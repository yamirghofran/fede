"""Orchestrate the full training-dataset generation pipeline.

Ties together:
    * ``SceneCorpus`` (corpus loading)
    * ``AsyncQueryGenerator`` (concurrent synthetic query creation — Types A, B, C)
    * ``PositiveAssigner`` (within-movie positive selection)
    * ``NegativeMiner`` (random negatives for round 1)

Exports to a line-delimited JSONL file with the schema::

    {"anchor": str, "positive": str, "negatives": [str, ...]}

Supports checkpoint / resume so that a long LLM-bound run can be
interrupted and continued without re-generating already-processed movies.

The build pipeline is fully async internally; ``build()`` is the public
synchronous entry point (it calls ``asyncio.run()``).  Movies are
processed ``LLM_CONCURRENCY`` at a time; the shared ``AsyncQueryGenerator``
rate-limiter caps global API throughput regardless of concurrency level.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from finetuning.config import (
    CHECKPOINT_INTERVAL,
    EMBEDDING_MODEL_ID,
    FINETUNING_DATA_DIR,
    FINETUNING_EMBED_DEVICE,
    LLM_CONCURRENCY,
    LLM_RATE_LIMIT_DELAY,
    QUERIES_PER_MOVIE_SYNOPSIS,
    RANDOM_NEGATIVES_PER_QUERY,
    TOP_SCENES_FOR_SUMMARY,
)
from finetuning.training.model import load_model
from finetuning.corpus.scene_corpus import MovieEntry, build_scene_corpus
from finetuning.dataset.negative_miner import sample_random_negatives
from finetuning.dataset.positive_assigner import PositiveAssigner
from finetuning.dataset.query_generator import (
    AsyncQueryGenerator,
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


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """End-to-end async dataset builder for round-1 training data.

    Usage::

        builder = DatasetBuilder(max_movies=1000)
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
        self._async_qgen: Optional[AsyncQueryGenerator] = None
        self._assigner: Optional[PositiveAssigner] = None

    # ----- lazy initialisation (heavy resources) -----

    def _ensure_corpus(self) -> Dict[str, MovieEntry]:
        if self._corpus is None:
            self._corpus = build_scene_corpus(max_movies=self._max_movies)
        return self._corpus

    def _ensure_async_query_gen(self) -> AsyncQueryGenerator:
        if self._async_qgen is None:
            self._async_qgen = AsyncQueryGenerator(api_key=self._api_key)
        return self._async_qgen

    def _ensure_assigner(self) -> PositiveAssigner:
        if self._assigner is None:
            logger.info("Loading embedding model %s on %s …", self._model_id, FINETUNING_EMBED_DEVICE)
            model = load_model(self._model_id, device=FINETUNING_EMBED_DEVICE)
            self._assigner = PositiveAssigner(model)
        return self._assigner

    # ----- per-movie async generation -----

    async def _generate_for_movie_async(
        self,
        movie_id: str,
        entry: MovieEntry,
        loop: asyncio.AbstractEventLoop,
    ) -> List[Dict[str, Any]]:
        """Generate all Type-A and Type-B rows for one movie, concurrently."""
        qgen = self._ensure_async_query_gen()
        assigner = self._ensure_assigner()
        corpus = self._ensure_corpus()
        rows: List[Dict[str, Any]] = []

        # -- Type A: synopsis queries (positive found via embedding similarity) --
        if entry.overview:
            synopsis_queries = await qgen.generate_synopsis_queries(
                entry.overview, entry.movie_title, n=QUERIES_PER_MOVIE_SYNOPSIS
            )
            if synopsis_queries:
                # Embedding encode is CPU-bound; offload so the event loop
                # stays free for other coroutines making API calls.
                matches = await loop.run_in_executor(
                    None, assigner.assign_batch, synopsis_queries, entry.scenes
                )
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

        # -- Type B: scene-summary queries (source scene IS the positive) --
        sorted_scenes = sorted(entry.scenes, key=lambda s: len(s.text), reverse=True)
        top_scenes = sorted_scenes[:TOP_SCENES_FOR_SUMMARY]
        scene_tasks = [qgen.generate_scene_summary(s.text, entry.movie_title) for s in top_scenes]
        summaries = await asyncio.gather(*scene_tasks, return_exceptions=True)
        for scene, summary in zip(top_scenes, summaries):
            if isinstance(summary, Exception) or not summary:
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

    # ----- paraphrase pass (Type C) — async -----

    async def _generate_paraphrases_async(
        self,
        rows: List[Dict[str, Any]],
        max_paraphrases: int = 2,
    ) -> List[Dict[str, Any]]:
        qgen = self._ensure_async_query_gen()
        corpus = self._ensure_corpus()

        async def _paraphrase_one(row: Dict[str, Any]) -> List[Dict[str, Any]]:
            paraphrases = await qgen.generate_paraphrases(
                row["anchor"], row["movie_title"], n=max_paraphrases
            )
            result = []
            for pq in paraphrases:
                negatives = sample_random_negatives(
                    row["movie_id"], corpus, n=RANDOM_NEGATIVES_PER_QUERY
                )
                result.append({
                    "anchor": pq,
                    "positive": row["positive"],
                    "negatives": negatives,
                    "movie_id": row["movie_id"],
                    "movie_title": row["movie_title"],
                    "query_type": "paraphrase",
                })
            return result

        semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

        async def _guarded(row: Dict[str, Any]) -> List[Dict[str, Any]]:
            async with semaphore:
                return await _paraphrase_one(row)

        all_results = await asyncio.gather(*[_guarded(r) for r in rows], return_exceptions=True)
        paraphrase_rows: List[Dict[str, Any]] = []
        for result in all_results:
            if isinstance(result, Exception):
                logger.warning("Paraphrase task failed: %s", result)
            else:
                paraphrase_rows.extend(result)
        return paraphrase_rows

    # ----- main async build loop -----

    async def _build_async(
        self,
        output: Path,
        resume: bool,
    ) -> Path:
        corpus = self._ensure_corpus()
        movie_ids = list(corpus.keys())
        loop = asyncio.get_event_loop()

        # Resume support
        processed: set = set()
        if resume:
            ckpt = load_checkpoint()
            if ckpt:
                processed = set(ckpt.get("processed_movies", []))
                logger.info("Resuming — %d movies already processed", len(processed))

        todo = [mid for mid in movie_ids if mid not in processed]
        logger.info(
            "%d movies to process (concurrency=%d, rate=%.1f req/s)",
            len(todo), LLM_CONCURRENCY, 1.0 / max(0.001, LLM_RATE_LIMIT_DELAY),
        )

        semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
        file_lock = asyncio.Lock()
        ab_pairs_written = 0
        completed_count = 0

        async def process_movie(idx: int, mid: str) -> int:
            """Returns number of rows written."""
            nonlocal completed_count, ab_pairs_written
            async with semaphore:
                entry = corpus[mid]
                logger.info(
                    "[%d/%d] Generating queries for %s",
                    idx + 1, len(movie_ids), entry.movie_title,
                )
                try:
                    rows = await self._generate_for_movie_async(mid, entry, loop)
                except Exception:
                    logger.error("Failed to generate rows for %s", mid, exc_info=True)
                    rows = []

            # File writes and counter updates are serialised via the lock
            # (though asyncio is single-threaded, explicit locking avoids
            # interleaving if we ever add thread-pool writers).
            async with file_lock:
                if rows:
                    _append_jsonl(output, rows)
                processed.add(mid)
                ab_pairs_written += len(rows)
                completed_count += 1
                if completed_count % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint({
                        "processed_movies": list(processed),
                        "ab_pairs": ab_pairs_written,
                    })
                    logger.info(
                        "Checkpoint saved — %d movies, %d pairs so far",
                        len(processed), ab_pairs_written,
                    )
            return len(rows)

        # Launch all movies; semaphore caps concurrency
        tasks = [process_movie(movie_ids.index(mid), mid) for mid in todo]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Phase 2: Type C paraphrases
        all_ab_rows = _read_jsonl(output)
        paraphrase_source = all_ab_rows[:len(all_ab_rows) // 3]
        del all_ab_rows
        logger.info("Generating paraphrases from %d source pairs …", len(paraphrase_source))
        paraphrase_rows = await self._generate_paraphrases_async(paraphrase_source)
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

    # ----- public synchronous entry point -----

    def build(
        self,
        output_path: Optional[Path] = None,
        resume: bool = True,
    ) -> Path:
        """Generate the full round-1 training dataset.

        Args:
            output_path: Where to write the JSONL.  Defaults to
                ``data/finetuning/training_pairs_r1.jsonl``.
            resume: If ``True``, skip movies already in the checkpoint.

        Returns:
            The path to the written JSONL file.
        """
        output = output_path or (FINETUNING_DATA_DIR / "training_pairs_r1.jsonl")
        return asyncio.run(self._build_async(output, resume))
