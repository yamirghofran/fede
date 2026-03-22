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

        has_overview = bool(entry.overview and entry.overview.strip())
        n_scenes = len(entry.scenes)
        if not has_overview and n_scenes == 0:
            logger.warning(
                "Movie '%s' has no overview and no scenes — nothing to generate",
                entry.movie_title,
            )
            return rows
        if not has_overview:
            logger.info("  No overview for '%s' — skipping synopsis queries", entry.movie_title)

        # Type A: synopsis queries
        if has_overview:
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

        total = len(movie_ids)
        start_offset = total - len(todo)

        for i, mid in enumerate(todo):
            entry = corpus[mid]
            logger.info(
                "[%d/%d] Generating queries for %s",
                start_offset + i + 1, total, entry.movie_title,
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
        resume: bool = True,
    ) -> Path:
        """Stage 2: Read raw queries, assign positives via embedding, sample negatives.

        Loads the embedding model, runs ``PositiveAssigner`` for Type A
        queries, and writes the final training-pair JSONL.

        Writes pairs incrementally per movie. Safe to interrupt and resume —
        already-processed movies (detected from the output file) are skipped.
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

        # Resume: detect movies already written to the output file
        done_movies: Set[str] = set()
        if resume and output.exists():
            done_movies = _movie_ids_in_jsonl(output)
            if done_movies:
                logger.info(
                    "Resuming Stage 2: %d movies already in %s — skipping them",
                    len(done_movies), output.name,
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

        todo_movies = {mid: rows for mid, rows in by_movie.items() if mid not in done_movies}
        logger.info(
            "Assembling pairs: %d queries across %d movies (%d already done)",
            sum(len(r) for r in todo_movies.values()), len(todo_movies), len(done_movies),
        )

        total_movies = len(by_movie)
        pairs_written = _count_lines(output)

        for idx, (movie_id, query_rows) in enumerate(todo_movies.items()):
            if movie_id not in corpus:
                logger.warning("Movie %s not in corpus — skipping", movie_id)
                continue
            entry = corpus[movie_id]

            movie_pairs: List[Dict[str, Any]] = []

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
                        movie_pairs.append({
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
                scene_idx = sr.get("scene_idx")
                if scene_idx is None or scene_idx >= len(top_scenes):
                    continue
                scene = top_scenes[scene_idx]
                negatives = sample_random_negatives(
                    movie_id, corpus, n=RANDOM_NEGATIVES_PER_QUERY,
                )
                movie_pairs.append({
                    "anchor": sr["query"],
                    "positive": scene.text,
                    "negatives": negatives,
                    "movie_id": movie_id,
                    "movie_title": entry.movie_title,
                    "query_type": "scene_summary",
                })

            # Write this movie's pairs immediately (incremental, resumable)
            if movie_pairs:
                _append_jsonl(output, movie_pairs)
                pairs_written += len(movie_pairs)

            if (idx + 1) % 50 == 0:
                logger.info(
                    "  [%d/%d] %d pairs written so far …",
                    len(done_movies) + idx + 1, total_movies, pairs_written,
                )

        logger.info("Stage 2 complete: %d training pairs → %s", pairs_written, output)

        # Update checkpoint
        all_done = done_movies | set(todo_movies.keys())
        save_checkpoint({
            "processed_movies": list(all_done),
            "ab_pairs": pairs_written,
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
