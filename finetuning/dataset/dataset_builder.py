"""Orchestrate the full training-dataset generation pipeline.

Ties together:
    * ``SceneCorpus`` (corpus loading)
    * ``QueryGenerator`` (synthetic query creation — Types A, B, C)
    * ``PositiveAssigner`` (within-movie positive selection)
    * ``NegativeMiner`` (random negatives for round 1)

Exports to a line-delimited JSONL file with the schema::

    {"anchor": str, "positive": str, "negatives": [str, ...]}

Supports checkpoint / resume so that a long LLM-bound run can be
interrupted and continued without re-generating already-processed movies.

The pipeline is fully synchronous.  With low-latency providers (e.g.
Gemini direct API, ~1-3 s/call) and a rate delay of 0.08 s, 1200 movies
complete in ~10-15 minutes sequentially.
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
from finetuning.training.model import load_model
from finetuning.corpus.scene_corpus import MovieEntry, build_scene_corpus
from finetuning.dataset.negative_miner import sample_random_negatives
from finetuning.dataset.positive_assigner import PositiveAssigner
from finetuning.dataset.query_generator import (
    QueryGenerator,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


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
    """End-to-end dataset builder for round-1 training data.

    Usage::

        builder = DatasetBuilder(max_movies=1200)
        builder.build(output_path=Path("data/finetuning/training_pairs_r1.jsonl"))
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
        self._query_gen: Optional[QueryGenerator] = None
        self._assigner: Optional[PositiveAssigner] = None

    # ----- lazy initialisation (heavy resources) -----

    def _ensure_corpus(self) -> Dict[str, MovieEntry]:
        if self._corpus is None:
            self._corpus = build_scene_corpus(max_movies=self._max_movies)
        return self._corpus

    def _ensure_query_gen(self) -> QueryGenerator:
        if self._query_gen is None:
            self._query_gen = QueryGenerator(api_key=self._api_key)
        return self._query_gen

    def _ensure_assigner(self) -> PositiveAssigner:
        if self._assigner is None:
            logger.info("Loading embedding model %s on %s …", self._model_id, FINETUNING_EMBED_DEVICE)
            model = load_model(self._model_id, device=FINETUNING_EMBED_DEVICE)
            self._assigner = PositiveAssigner(model)
        return self._assigner

    # ----- per-movie generation -----

    def _generate_for_movie(
        self,
        movie_id: str,
        entry: MovieEntry,
    ) -> List[Dict[str, Any]]:
        """Generate all Type-A and Type-B rows for one movie."""
        qgen = self._ensure_query_gen()
        assigner = self._ensure_assigner()
        corpus = self._ensure_corpus()
        rows: List[Dict[str, Any]] = []

        # -- Type A: synopsis queries --
        if entry.overview:
            synopsis_queries = qgen.generate_synopsis_queries(
                entry.overview, entry.movie_title, n=QUERIES_PER_MOVIE_SYNOPSIS
            )
            qgen.throttle()

            if synopsis_queries:
                matches = assigner.assign_batch(synopsis_queries, entry.scenes)
                for match in matches:
                    negatives = sample_random_negatives(movie_id, corpus, n=RANDOM_NEGATIVES_PER_QUERY)
                    for pos in match.positives:
                        rows.append({
                            "anchor": match.query,
                            "positive": pos.text,
                            "negatives": negatives,
                            "movie_id": movie_id,
                            "movie_title": entry.movie_title,
                            "query_type": "synopsis",
                        })

        # -- Type B: scene-summary queries --
        sorted_scenes = sorted(entry.scenes, key=lambda s: len(s.text), reverse=True)
        top_scenes = sorted_scenes[:TOP_SCENES_FOR_SUMMARY]
        for scene in top_scenes:
            summary = qgen.generate_scene_summary(scene.text, entry.movie_title)
            qgen.throttle()
            if not summary:
                continue
            negatives = sample_random_negatives(movie_id, corpus, n=RANDOM_NEGATIVES_PER_QUERY)
            rows.append({
                "anchor": summary,
                "positive": scene.text,
                "negatives": negatives,
                "movie_id": movie_id,
                "movie_title": entry.movie_title,
                "query_type": "scene_summary",
            })

        return rows

    # ----- paraphrase pass (Type C) -----

    def _generate_paraphrases(
        self,
        rows: List[Dict[str, Any]],
        max_paraphrases: int = 2,
    ) -> List[Dict[str, Any]]:
        qgen = self._ensure_query_gen()
        corpus = self._ensure_corpus()
        paraphrase_rows: List[Dict[str, Any]] = []

        for i, row in enumerate(rows, 1):
            paraphrases = qgen.generate_paraphrases(
                row["anchor"], row["movie_title"], n=max_paraphrases
            )
            qgen.throttle()
            for pq in paraphrases:
                negatives = sample_random_negatives(
                    row["movie_id"], corpus, n=RANDOM_NEGATIVES_PER_QUERY
                )
                paraphrase_rows.append({
                    "anchor": pq,
                    "positive": row["positive"],
                    "negatives": negatives,
                    "movie_id": row["movie_id"],
                    "movie_title": row["movie_title"],
                    "query_type": "paraphrase",
                })
            if i % 100 == 0:
                logger.info("  Paraphrased %d / %d source pairs", i, len(rows))

        return paraphrase_rows

    # ----- main entry point -----

    def build(
        self,
        output_path: Optional[Path] = None,
        resume: bool = True,
    ) -> Path:
        """Generate the full round-1 training dataset.

        Args:
            output_path: Where to write the JSONL.  Defaults to
                ``data/finetuning/training_pairs_r1.jsonl``.
            resume: If ``True``, skip movies already in the checkpoint
                *and* movies already present in the output JSONL (prevents
                duplicates when a run is interrupted between checkpoints).

        Returns:
            The path to the written JSONL file.
        """
        output = output_path or (FINETUNING_DATA_DIR / "training_pairs_r1.jsonl")
        corpus = self._ensure_corpus()
        movie_ids = list(corpus.keys())

        # Resume support — union checkpoint AND JSONL to prevent dupes
        processed: set = set()
        if resume:
            ckpt = load_checkpoint()
            if ckpt:
                processed = set(ckpt.get("processed_movies", []))
                logger.info("Checkpoint: %d movies", len(processed))

            # Guard against the gap between last checkpoint and actual JSONL
            already_in_file = _movie_ids_in_jsonl(output)
            if already_in_file - processed:
                logger.info(
                    "Found %d extra movies in JSONL not in checkpoint — adding to skip set",
                    len(already_in_file - processed),
                )
            processed |= already_in_file

        todo = [mid for mid in movie_ids if mid not in processed]
        logger.info("%d movies to process (%d already done)", len(todo), len(processed))

        # Phase 1: Type A + B for each movie
        ab_pairs_written = _count_lines(output)
        for i, mid in enumerate(todo):
            entry = corpus[mid]
            logger.info(
                "[%d/%d] Generating queries for %s",
                len(processed) + i + 1, len(movie_ids), entry.movie_title,
            )

            rows = self._generate_for_movie(mid, entry)

            if rows:
                _append_jsonl(output, rows)
                ab_pairs_written += len(rows)
                processed.add(mid)
            else:
                logger.warning("No pairs generated for %s — will retry on next resume", entry.movie_title)

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint({"processed_movies": list(processed), "ab_pairs": ab_pairs_written})
                logger.info("Checkpoint saved — %d movies, %d pairs so far", len(processed), ab_pairs_written)

        # Save checkpoint after Phase 1 completes
        save_checkpoint({"processed_movies": list(processed), "ab_pairs": ab_pairs_written})

        # Phase 2: Type C paraphrases
        all_ab_rows = _read_jsonl(output)
        paraphrase_source = all_ab_rows[:len(all_ab_rows) // 3]
        del all_ab_rows
        logger.info("Generating paraphrases from %d source pairs …", len(paraphrase_source))
        paraphrase_rows = self._generate_paraphrases(paraphrase_source)
        _append_jsonl(output, paraphrase_rows)

        total = _count_lines(output)
        logger.info("Dataset complete: %d total pairs written to %s", total, output)
        save_checkpoint({
            "processed_movies": list(processed),
            "ab_pairs": ab_pairs_written,
            "paraphrase_pairs": len(paraphrase_rows),
            "total_pairs": total,
            "status": "complete",
        })

        return output
