"""Batch-based dataset generation using the Gemini Batch API.

Instead of thousands of sequential synchronous API calls, this module:

1. Builds all prompts locally as a JSONL file.
2. Uploads to Google and submits a single batch job.
3. Polls until complete.
4. Downloads results and post-processes them into training pairs.

The Batch API runs at 50% of standard cost, has no per-minute rate limits,
and typically completes in minutes for datasets of this size.

Usage::

    from finetuning.dataset.batch_builder import BatchDatasetBuilder

    builder = BatchDatasetBuilder(max_movies=1200)
    # Step 1: build prompts + submit
    job_name = builder.build_and_submit()
    # Step 2: poll (can be called later / after kernel restart)
    builder.poll_until_done(job_name)
    # Step 3: post-process into training pairs
    output = builder.process_results(job_name)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from google import genai
from google.genai import types

from finetuning.config import (
    EMBEDDING_MODEL_ID,
    FINETUNING_DATA_DIR,
    FINETUNING_EMBED_DEVICE,
    GEMINI_BATCH_MODEL,
    GEMINI_BATCH_POLL_INTERVAL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    QUERIES_PER_MOVIE_SYNOPSIS,
    RANDOM_NEGATIVES_PER_QUERY,
    TOP_SCENES_FOR_SUMMARY,
)
from finetuning.corpus.scene_corpus import MovieEntry, build_scene_corpus
from finetuning.dataset.negative_miner import sample_random_negatives
from finetuning.dataset.query_generator import (
    check_leakage,
    _parse_json_string_list,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (same text as query_generator.py, adapted for batch)
# ---------------------------------------------------------------------------

_SYNOPSIS_PROMPT = """\
You generate training data for a semantic search engine over movie scenes.

Task: Read the synopsis and write {n} distinct short queries a real user might type when searching for scenes from this kind of film.

Requirements for each query:
- One sentence, natural web-search style (not a critic's logline or marketing blurb).
- Under ~25 words.
- Spread angles across the set: e.g. plot situation, central conflict, relationship dynamic, theme or tone — avoid near-duplicates.
- Use only generic wording: do NOT use the film title, character names, actor names, or other proper nouns taken from the synopsis.
- No dialogue, no quotation marks, no "movie about" meta-phrasing unless unavoidable.

Synopsis:
{synopsis}"""

_SCENE_SUMMARY_PROMPT = """\
You are creating training data for a movie retrieval system.

Convert the following movie script scene into ONE realistic user search query.

The query should describe the most distinctive event, conflict, relationship, or revelation in the scene as a person might remember it when searching for a movie.

Rules:
- Exactly one sentence.
- Plain natural English.
- No screenplay formatting.
- No movie title.
- No character names.
- No actor names.
- No direct quotes.
- No copied phrases unless unavoidable.
- No invented details.
- Focus on the core narrative content, not surface visuals.
- Make it distinctive but natural.
- Avoid spoilers unless necessary to preserve the meaning of the scene.

If the scene is too generic to form a useful retrieval query, set is_usable to false.

Scene:
{scene_text}"""

# ---------------------------------------------------------------------------
# Structured output schemas for Gemini Batch API
# ---------------------------------------------------------------------------

_SYNOPSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "queries": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        }
    },
    "required": ["queries"],
}

_SCENE_SUMMARY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "query": {"type": "STRING"},
        "is_usable": {"type": "BOOLEAN"},
        "contains_title": {"type": "BOOLEAN"},
        "contains_character_name": {"type": "BOOLEAN"},
        "too_generic": {"type": "BOOLEAN"},
    },
    "required": ["query", "is_usable", "contains_title", "contains_character_name", "too_generic"],
}


# ---------------------------------------------------------------------------
# Batch request builder
# ---------------------------------------------------------------------------

def _make_request(
    prompt: str,
    schema: dict,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> dict:
    """Build a single GenerateContentRequest dict for the batch JSONL."""
    return {
        "contents": [{"parts": [{"text": prompt}], "role": "user"}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }


class BatchDatasetBuilder:
    """Builds the training dataset via the Gemini Batch API.

    Separates the pipeline into three independently resumable stages:
    build+submit, poll, process.
    """

    def __init__(
        self,
        max_movies: Optional[int] = None,
        model: str = GEMINI_BATCH_MODEL,
    ) -> None:
        self._max_movies = max_movies
        self._model = model

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        self._client = genai.Client(api_key=api_key)

        self._corpus: Optional[Dict[str, MovieEntry]] = None
        self._requests_path = FINETUNING_DATA_DIR / "batch_requests.jsonl"
        self._results_path = FINETUNING_DATA_DIR / "batch_results.jsonl"

    def _ensure_corpus(self) -> Dict[str, MovieEntry]:
        if self._corpus is None:
            self._corpus = build_scene_corpus(max_movies=self._max_movies)
        return self._corpus

    # ---------------------------------------------------------------
    # Stage 1: Build prompt JSONL + submit batch job
    # ---------------------------------------------------------------

    def build_batch_requests(self) -> Path:
        """Build a JSONL file of all Type A + Type B prompts.

        Returns the path to the written JSONL file.
        """
        corpus = self._ensure_corpus()
        self._requests_path.parent.mkdir(parents=True, exist_ok=True)

        n_synopsis = 0
        n_scene = 0

        with open(self._requests_path, "w", encoding="utf-8") as f:
            for movie_id, entry in corpus.items():
                # Type A: synopsis queries
                if entry.overview and entry.overview.strip():
                    prompt = _SYNOPSIS_PROMPT.format(
                        n=QUERIES_PER_MOVIE_SYNOPSIS, synopsis=entry.overview,
                    )
                    req = {
                        "key": f"synopsis__{movie_id}",
                        "request": _make_request(prompt, _SYNOPSIS_SCHEMA),
                    }
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
                    n_synopsis += 1

                # Type B: scene summaries (top N longest scenes)
                sorted_scenes = sorted(entry.scenes, key=lambda s: len(s.text), reverse=True)
                for i, scene in enumerate(sorted_scenes[:TOP_SCENES_FOR_SUMMARY]):
                    prompt = _SCENE_SUMMARY_PROMPT.format(scene_text=scene.text)
                    req = {
                        "key": f"scene__{movie_id}__{i}",
                        "request": _make_request(prompt, _SCENE_SUMMARY_SCHEMA),
                    }
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
                    n_scene += 1

        logger.info(
            "Batch JSONL built: %d synopsis + %d scene prompts → %s",
            n_synopsis, n_scene, self._requests_path,
        )
        return self._requests_path

    def submit_batch_job(self, requests_path: Optional[Path] = None) -> str:
        """Upload the JSONL and submit the batch job.

        Returns the batch job name (e.g. 'batches/123456').
        """
        path = requests_path or self._requests_path
        if not path.exists():
            raise FileNotFoundError(f"Batch requests file not found: {path}")

        logger.info("Uploading %s to Gemini Files API …", path)
        uploaded = self._client.files.upload(
            file=str(path),
            config=types.UploadFileConfig(
                display_name="fede-batch-requests",
                mime_type="jsonl",
            ),
        )
        logger.info("Uploaded: %s", uploaded.name)

        logger.info("Submitting batch job (model=%s) …", self._model)
        job = self._client.batches.create(
            model=self._model,
            src=uploaded.name,
            config={"display_name": "fede-training-data"},
        )
        logger.info("Batch job created: %s", job.name)
        return job.name

    def build_and_submit(self) -> str:
        """Build prompts and submit in one call. Returns the job name."""
        self.build_batch_requests()
        return self.submit_batch_job()

    # ---------------------------------------------------------------
    # Stage 2: Poll until done
    # ---------------------------------------------------------------

    def poll_until_done(
        self,
        job_name: str,
        poll_interval: int = GEMINI_BATCH_POLL_INTERVAL,
    ) -> str:
        """Poll the batch job until it reaches a terminal state.

        Returns the final state name (e.g. 'JOB_STATE_SUCCEEDED').
        """
        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

        logger.info("Polling batch job %s every %ds …", job_name, poll_interval)
        while True:
            job = self._client.batches.get(name=job_name)
            state = job.state.name if hasattr(job.state, "name") else str(job.state)
            if state in terminal:
                logger.info("Batch job %s → %s", job_name, state)
                if state == "JOB_STATE_FAILED":
                    logger.error("Batch job failed: %s", getattr(job, "error", "unknown"))
                return state
            logger.info("  state=%s — waiting %ds …", state, poll_interval)
            time.sleep(poll_interval)

    # ---------------------------------------------------------------
    # Stage 3: Download + post-process results
    # ---------------------------------------------------------------

    def download_results(self, job_name: str) -> Path:
        """Download results JSONL from a completed batch job."""
        job = self._client.batches.get(name=job_name)

        self._results_path.parent.mkdir(parents=True, exist_ok=True)

        # Inline responses (small jobs)
        if job.dest and job.dest.inlined_responses:
            logger.info("Extracting inline responses …")
            with open(self._results_path, "w", encoding="utf-8") as f:
                for resp in job.dest.inlined_responses:
                    row: dict = {}
                    if resp.response and resp.response.text:
                        row["response"] = {"text": resp.response.text}
                    if resp.error:
                        row["error"] = str(resp.error)
                    meta = resp.metadata or {}
                    if "key" in meta:
                        row["key"] = meta["key"]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # File-based responses (larger jobs)
        elif job.dest and job.dest.file_name:
            logger.info("Downloading result file %s …", job.dest.file_name)
            content_bytes = self._client.files.download(file=job.dest.file_name)
            with open(self._results_path, "wb") as f:
                f.write(content_bytes)
        else:
            raise RuntimeError(f"No results found for job {job_name}")

        n_lines = sum(1 for _ in open(self._results_path))
        logger.info("Results downloaded: %d lines → %s", n_lines, self._results_path)
        return self._results_path

    def process_results(
        self,
        job_name: Optional[str] = None,
        results_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Parse batch results and build training pairs.

        If *job_name* is provided, downloads results first.
        If *results_path* is provided, skips download and uses the file directly.
        """
        if job_name and not results_path:
            results_path = self.download_results(job_name)
        elif not results_path:
            results_path = self._results_path
            if not results_path.exists():
                raise FileNotFoundError(
                    "No results file found. Provide job_name or results_path."
                )

        output = output_path or (FINETUNING_DATA_DIR / "training_pairs_r1.jsonl")
        corpus = self._ensure_corpus()

        # Load embedding model for PositiveAssigner (Type A only)
        from finetuning.training.model import load_model
        from finetuning.dataset.positive_assigner import PositiveAssigner

        logger.info("Loading embedding model for positive assignment …")
        emb_model = load_model(EMBEDDING_MODEL_ID, device=FINETUNING_EMBED_DEVICE)
        assigner = PositiveAssigner(emb_model)

        # Parse all results keyed by request key
        raw_results: Dict[str, dict] = {}
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = row.get("key", "")
                if not key:
                    continue
                raw_results[key] = row

        logger.info("Parsed %d batch results", len(raw_results))

        # Group synopsis results by movie for batch positive assignment
        synopsis_by_movie: Dict[str, List[str]] = {}
        scene_results: List[dict] = []

        for key, row in raw_results.items():
            text = _extract_text(row)
            if not text:
                continue

            if key.startswith("synopsis__"):
                movie_id = key.split("__", 1)[1]
                queries = _parse_synopsis_response(text, movie_id, corpus)
                if queries:
                    synopsis_by_movie.setdefault(movie_id, []).extend(queries)
            elif key.startswith("scene__"):
                parts = key.split("__")
                movie_id = parts[1]
                scene_idx = int(parts[2])
                scene_results.append({
                    "movie_id": movie_id,
                    "scene_idx": scene_idx,
                    "raw": text,
                })

        # Build training pairs
        output.parent.mkdir(parents=True, exist_ok=True)
        pairs: List[Dict[str, Any]] = []

        # Type A: synopsis queries → assign best scene via embedding similarity
        for movie_id, queries in synopsis_by_movie.items():
            if movie_id not in corpus:
                continue
            entry = corpus[movie_id]
            matches = assigner.assign_batch(queries, entry.scenes)
            for match in matches:
                negatives = sample_random_negatives(movie_id, corpus, n=RANDOM_NEGATIVES_PER_QUERY)
                for pos in match.positives:
                    pairs.append({
                        "anchor": match.query,
                        "positive": pos.text,
                        "negatives": negatives,
                        "movie_id": movie_id,
                        "movie_title": entry.movie_title,
                        "query_type": "synopsis",
                    })

        logger.info("Type A pairs: %d", len(pairs))
        n_type_a = len(pairs)

        # Type B: scene summaries → source scene is the positive
        for sr in scene_results:
            movie_id = sr["movie_id"]
            if movie_id not in corpus:
                continue
            entry = corpus[movie_id]
            result = _parse_scene_response(sr["raw"], entry.movie_title)
            if not result:
                continue

            sorted_scenes = sorted(entry.scenes, key=lambda s: len(s.text), reverse=True)
            top_scenes = sorted_scenes[:TOP_SCENES_FOR_SUMMARY]
            idx = sr["scene_idx"]
            if idx >= len(top_scenes):
                continue

            scene = top_scenes[idx]
            negatives = sample_random_negatives(movie_id, corpus, n=RANDOM_NEGATIVES_PER_QUERY)
            pairs.append({
                "anchor": result,
                "positive": scene.text,
                "negatives": negatives,
                "movie_id": movie_id,
                "movie_title": entry.movie_title,
                "query_type": "scene_summary",
            })

        logger.info("Type B pairs: %d", len(pairs) - n_type_a)

        # Write all pairs
        with open(output, "w", encoding="utf-8") as f:
            for row in pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info("Training pairs written: %d → %s", len(pairs), output)

        # Save checkpoint
        processed_movies: Set[str] = set()
        for p in pairs:
            processed_movies.add(p["movie_id"])
        save_checkpoint({
            "processed_movies": list(processed_movies),
            "ab_pairs": len(pairs),
            "status": "batch_complete",
        })

        return output


# ---------------------------------------------------------------------------
# Result parsing helpers
# ---------------------------------------------------------------------------

def _extract_text(row: dict) -> Optional[str]:
    """Extract the text content from a batch result row."""
    if "error" in row and row["error"]:
        return None
    resp = row.get("response", {})
    if isinstance(resp, dict):
        text = resp.get("text", "")
        if text:
            return text
        # Nested candidates format from file-based results
        candidates = resp.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
    return None


def _parse_synopsis_response(
    text: str,
    movie_id: str,
    corpus: Dict[str, MovieEntry],
) -> List[str]:
    """Parse Type A synopsis queries from structured JSON response."""
    queries = _parse_json_string_list(text, preferred_key="queries")
    if not queries:
        return []

    entry = corpus.get(movie_id)
    if not entry:
        return []

    return [q for q in queries if not check_leakage(q, entry.movie_title)]


def _parse_scene_response(text: str, movie_title: str) -> Optional[str]:
    """Parse Type B scene summary from structured JSON response."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to treating the text as a raw query
        query = text.strip().strip('"').strip("'")
        if query.upper() == "SKIP" or not query:
            return None
        if check_leakage(query, movie_title):
            return None
        return query

    if not isinstance(parsed, dict):
        return None

    if not parsed.get("is_usable", True):
        return None
    if parsed.get("contains_title", False):
        return None
    if parsed.get("contains_character_name", False):
        return None
    if parsed.get("too_generic", False):
        return None

    query = parsed.get("query", "").strip()
    if not query:
        return None

    # Double-check with our own leakage detector
    if check_leakage(query, movie_title):
        return None

    return query
