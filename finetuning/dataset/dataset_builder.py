"""Orchestrate the full training-dataset generation pipeline.

The pipeline is split into two independently runnable stages:

**Stage 1 — generate_queries()** (LLM-only, no embedding model):
    Iterates every movie in the corpus, calls the LLM for Type A (synopsis)
    and Type B (scene summary) queries, and writes raw results to
    ``data/finetuning/raw_queries.jsonl``.  Checkpoints every 50 movies;
    safe to interrupt and resume.

**Stage 2 — assemble_pairs()** (local compute, no API calls):
    Reads ``raw_queries.jsonl``, loads the embedding model, runs
    ``PositiveAssigner`` for Type A queries, samples random negatives,
    and writes the final ``training_pairs_r1.jsonl``.

``build()`` is a convenience method that runs both stages back-to-back.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from finetuning.config import (
    CHECKPOINT_INTERVAL,
    EMBEDDING_MODEL_ID,
    FINETUNING_DATA_DIR,
    FINETUNING_EMBED_DEVICE,
    QUERIES_PER_MOVIE_SYNOPSIS,
    RANDOM_NEGATIVES_PER_QUERY,
    TOP_SCENES_FOR_SUMMARY,
)
from finetuning.corpus.scene_corpus import MovieEntry, build_scene_corpus
from finetuning.dataset.query_generator import (
    QueryGenerator,
    check_leakage,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

_RAW_QUERIES_DEFAULT = FINETUNING_DATA_DIR / "raw_queries.jsonl"
_TRAINING_PAIRS_DEFAULT = FINETUNING_DATA_DIR / "training_pairs_r1.jsonl"


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _movie_ids_in_jsonl(path: Path) -> Set[str]:
    """Scan existing JSONL for movie_ids already written (guards against dupes)."""
    ids: Set[str] = set()
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["movie_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """Two-stage dataset builder for round-1 training data.

    Stage 1 (``generate_queries``) — LLM calls only, no embedding model.
    Stage 2 (``assemble_pairs``) — local embedding + pair assembly, no API.
    ``build()`` runs both in sequence.
    """

    def __init__(
        self,
        max_movies: Optional[int] = None,
        model_id: str = EMBEDDING_MODEL_ID,
        api_key: Optional[str] = None,
    ) -> None:
        self._max_movies = max_movies
        self._model_id = model_id
        self._api_key = api_key
        self._corpus: Optional[Dict[str, MovieEntry]] = None

    def _ensure_corpus(self) -> Dict[str, MovieEntry]:
        if self._corpus is None:
            self._corpus = build_scene_corpus(max_movies=self._max_movies)
        return self._corpus

    # =======================================================================
    # Stage 1: LLM-only query generation
    # =======================================================================

    def _generate_queries_for_movie(
        self,
        movie_id: str,
        entry: MovieEntry,
        qgen: QueryGenerator,
    ) -> List[Dict[str, Any]]:
        """Generate raw query rows for one movie (no embedding needed)."""
        rows: List[Dict[str, Any]] = []

        # Type A: synopsis queries
        if entry.overview:
            queries = qgen.generate_synopsis_queries(
                entry.overview, entry.movie_title, n=QUERIES_PER_MOVIE_SYNOPSIS,
            )
            qgen.throttle()
            for q in queries:
                rows.append({
                    "movie_id": movie_id,
                    "movie_title": entry.movie_title,
                    "query": q,
                    "query_type": "synopsis",
                    "scene_idx": None,
                })

        # Type B: scene summaries (top N longest scenes)
        sorted_scenes = sorted(entry.scenes, key=lambda s: len(s.text), reverse=True)
        top_scenes = sorted_scenes[:TOP_SCENES_FOR_SUMMARY]
        for i, scene in enumerate(top_scenes):
            summary = qgen.generate_scene_summary(scene.text, entry.movie_title)
            qgen.throttle()
            if not summary:
                continue
            rows.append({
                "movie_id": movie_id,
                "movie_title": entry.movie_title,
                "query": summary,
                "query_type": "scene_summary",
                "scene_idx": i,
            })

        return rows

    def generate_queries(
        self,
        output_path: Optional[Path] = None,
        resume: bool = True,
    ) -> Path:
        """Stage 1: Generate raw queries via LLM (no embedding model loaded).

        Writes one JSONL line per query::

            {"movie_id": "...", "movie_title": "...", "query": "...",
             "query_type": "synopsis|scene_summary", "scene_idx": null|int}

        Checkpoints every 50 movies. Safe to interrupt and resume.
        """
        output = output_path or _RAW_QUERIES_DEFAULT
        corpus = self._ensure_corpus()
        movie_ids = list(corpus.keys())

        # Resume support
        processed: set = set()
        if resume:
            ckpt = load_checkpoint()
            if ckpt:
                processed = set(ckpt.get("processed_movies", []))
                logger.info("Checkpoint: %d movies", len(processed))
            already_in_file = _movie_ids_in_jsonl(output)
            if already_in_file - processed:
                logger.info(
                    "Found %d extra movies in JSONL not in checkpoint — adding to skip set",
                    len(already_in_file - processed),
                )
            processed |= already_in_file

        todo = [mid for mid in movie_ids if mid not in processed]
        logger.info("%d movies to process (%d already done)", len(todo), len(processed))

        qgen = QueryGenerator(api_key=self._api_key)
        queries_written = _count_lines(output)

        for i, mid in enumerate(todo):
            entry = corpus[mid]
            logger.info(
                "[%d/%d] Generating queries for %s",
                len(processed) + i + 1, len(movie_ids), entry.movie_title,
            )

            rows = self._generate_queries_for_movie(mid, entry, qgen)

            if rows:
                _append_jsonl(output, rows)
                queries_written += len(rows)
                processed.add(mid)
            else:
                logger.warning(
                    "No queries generated for %s — will retry on next resume",
                    entry.movie_title,
                )

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint({
                    "processed_movies": list(processed),
                    "raw_queries": queries_written,
                })
                logger.info(
                    "Checkpoint saved — %d movies, %d queries so far",
                    len(processed), queries_written,
                )

        save_checkpoint({
            "processed_movies": list(processed),
            "raw_queries": queries_written,
            "status": "queries_complete",
        })
        logger.info(
            "Stage 1 complete: %d queries from %d movies → %s",
            queries_written, len(processed), output,
        )
        return output

    # =======================================================================
    # Stage 2: Local embedding + pair assembly (no API calls)
    # =======================================================================

    def assemble_pairs(
        self,
        queries_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Stage 2: Read raw queries, assign positives via embedding, sample negatives.

        Loads the embedding model, runs ``PositiveAssigner`` for Type A
        queries, and writes the final training-pair JSONL.
        """
        from finetuning.training.model import load_model
        from finetuning.dataset.positive_assigner import PositiveAssigner
        from finetuning.dataset.negative_miner import sample_random_negatives

        queries_path = queries_path or _RAW_QUERIES_DEFAULT
        output = output_path or _TRAINING_PAIRS_DEFAULT
        corpus = self._ensure_corpus()

        if not queries_path.exists():
            raise FileNotFoundError(
                f"Raw queries file not found: {queries_path}. Run generate_queries() first."
            )

        # Load embedding model
        logger.info("Loading embedding model %s on %s …", self._model_id, FINETUNING_EMBED_DEVICE)
        emb_model = load_model(self._model_id, device=FINETUNING_EMBED_DEVICE)
        assigner = PositiveAssigner(emb_model)

        # Group raw queries by movie
        raw_rows = _read_jsonl(queries_path)
        by_movie: Dict[str, List[Dict[str, Any]]] = {}
        for row in raw_rows:
            by_movie.setdefault(row["movie_id"], []).append(row)

        logger.info(
            "Assembling pairs: %d queries across %d movies",
            len(raw_rows), len(by_movie),
        )

        pairs: List[Dict[str, Any]] = []

        for movie_id, query_rows in by_movie.items():
            if movie_id not in corpus:
                logger.warning("Movie %s not in corpus — skipping", movie_id)
                continue
            entry = corpus[movie_id]

            # Separate Type A (synopsis) and Type B (scene_summary) queries
            synopsis_queries = [r["query"] for r in query_rows if r["query_type"] == "synopsis"]
            scene_rows = [r for r in query_rows if r["query_type"] == "scene_summary"]

            # Type A: embed all scenes, find best positive per query
            if synopsis_queries:
                matches = assigner.assign_batch(synopsis_queries, entry.scenes)
                for match in matches:
                    negatives = sample_random_negatives(
                        movie_id, corpus, n=RANDOM_NEGATIVES_PER_QUERY,
                    )
                    for pos in match.positives:
                        pairs.append({
                            "anchor": match.query,
                            "positive": pos.text,
                            "negatives": negatives,
                            "movie_id": movie_id,
                            "movie_title": entry.movie_title,
                            "query_type": "synopsis",
                        })

            # Type B: source scene is the positive (indexed by scene_idx)
            sorted_scenes = sorted(entry.scenes, key=lambda s: len(s.text), reverse=True)
            top_scenes = sorted_scenes[:TOP_SCENES_FOR_SUMMARY]
            for sr in scene_rows:
                idx = sr.get("scene_idx")
                if idx is None or idx >= len(top_scenes):
                    continue
                scene = top_scenes[idx]
                negatives = sample_random_negatives(
                    movie_id, corpus, n=RANDOM_NEGATIVES_PER_QUERY,
                )
                pairs.append({
                    "anchor": sr["query"],
                    "positive": scene.text,
                    "negatives": negatives,
                    "movie_id": movie_id,
                    "movie_title": entry.movie_title,
                    "query_type": "scene_summary",
                })

            if len(pairs) % 500 == 0 and pairs:
                logger.info("  %d pairs assembled so far …", len(pairs))

        # Write all pairs
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for row in pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info("Stage 2 complete: %d training pairs → %s", len(pairs), output)

        # Update checkpoint
        processed_movies = {p["movie_id"] for p in pairs}
        save_checkpoint({
            "processed_movies": list(processed_movies),
            "ab_pairs": len(pairs),
            "status": "complete",
        })

        return output

    # =======================================================================
    # Convenience: run both stages
    # =======================================================================

    def build(
        self,
        output_path: Optional[Path] = None,
        resume: bool = True,
    ) -> Path:
        """Run Stage 1 (generate_queries) then Stage 2 (assemble_pairs)."""
        queries_path = self.generate_queries(resume=resume)
        return self.assemble_pairs(
            queries_path=queries_path,
            output_path=output_path,
        )
