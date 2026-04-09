"""Generate held-out evaluation query sets for movie-level retrieval.

Two generators are provided:

* ``generate_eval_dataset`` — **overview-based** (Type A): one whole-movie
  description per TMDB overview.  Tests generalisation to abstract queries.
* ``generate_scene_eval_dataset`` — **scene-based** (Type B): queries
  generated from actual scene text, directly aligned with the scene-level
  training objective.  This is the recommended primary evaluation.

Output schema (``data/finetuning/eval_queries.json``)::

    [
      {"query": str, "movie_id": str, "movie_title": str},
      ...
    ]
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from finetuning.config import (
    EVAL_DATASET_SIZE,
    FINETUNING_DATA_DIR,
    TOP_SCENES_FOR_SUMMARY,
)
from finetuning.corpus.scene_corpus import MovieEntry
from finetuning.dataset.query_generator import QueryGenerator, check_leakage

logger = logging.getLogger(__name__)

_EVAL_PROMPT = """\
You are generating an evaluation query for a movie search system.

Given the following movie synopsis, generate ONE descriptive sentence that
captures the essence, plot, themes, or atmosphere of the movie.

Rules:
- Do not mention the movie title.
- Do not mention character names.
- Do not mention actor names.
- Do not include dates or locations that directly identify the movie.
- The sentence should be complete, grammatically correct, and natural.

Return ONLY the single sentence, no other text.

Synopsis:
{synopsis}"""


def generate_eval_dataset(
    corpus_movie_ids: Set[str],
    metadata: Dict[str, Any],
    output_path: Optional[Path] = None,
    n: int = EVAL_DATASET_SIZE,
    api_key: Optional[str] = None,
    full_corpus_movie_ids: Optional[Set[str]] = None,
) -> Path:
    """Generate a held-out evaluation dataset.

    Args:
        corpus_movie_ids: The set of ``movie_id`` values already used for
            training.  These movies are excluded from the eval set.
        metadata: The full ``clean_parsed_meta.json`` dict.
        output_path: Where to write the JSON.  Defaults to
            ``data/finetuning/eval_queries.json``.
        n: Target number of evaluation queries.
        api_key: OpenRouter API key override.
        full_corpus_movie_ids: If provided, candidates are restricted to
            movies whose ``movie_id`` is in this set (i.e. movies that
            have parseable tagged scripts).  This prevents generating
            eval queries for movies that have no scenes in the corpus.

    Returns:
        The path to the saved JSON file.
    """
    output = output_path or (FINETUNING_DATA_DIR / "eval_queries.json")
    qgen = QueryGenerator(api_key=api_key)

    candidates: List[Dict[str, Any]] = []
    for key, entry in metadata.items():
        file_info = entry.get("file", {})
        tmdb_info = entry.get("tmdb", {})
        movie_title = file_info.get("name", key)
        movie_id = movie_title.lower().replace(" ", "_").replace("-", "_")

        if movie_id in corpus_movie_ids:
            continue

        if full_corpus_movie_ids is not None and movie_id not in full_corpus_movie_ids:
            continue

        overview = tmdb_info.get("overview", "")
        if not overview or len(overview.split()) < 10:
            continue

        candidates.append({
            "movie_id": movie_id,
            "movie_title": movie_title,
            "overview": overview,
        })

    random.shuffle(candidates)
    candidates = candidates[:n * 2]  # over-sample to account for failures

    logger.info("Generating eval queries from %d candidate movies (target: %d)", len(candidates), n)

    eval_queries: List[Dict[str, str]] = []
    for cand in candidates:
        if len(eval_queries) >= n:
            break

        prompt = _EVAL_PROMPT.format(synopsis=cand["overview"])
        raw = qgen._call_llm(prompt)
        qgen.throttle()

        if raw is None:
            continue

        query = raw.strip().strip('"').strip("'")
        if check_leakage(query, cand["movie_title"]):
            logger.debug("Leakage detected for %s — skipping", cand["movie_title"])
            continue

        eval_queries.append({
            "query": query,
            "movie_id": cand["movie_id"],
            "movie_title": cand["movie_title"],
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(eval_queries, f, ensure_ascii=False, indent=2)

    logger.info("Eval dataset saved: %d queries → %s", len(eval_queries), output)
    return output


# ---------------------------------------------------------------------------
# Scene-based evaluation dataset (recommended)
# ---------------------------------------------------------------------------

def generate_scene_eval_dataset(
    train_movie_ids: Set[str],
    corpus: Dict[str, MovieEntry],
    output_path: Optional[Path] = None,
    scenes_per_movie: int = TOP_SCENES_FOR_SUMMARY,
    api_key: Optional[str] = None,
    seed: int = 42,
) -> Path:
    """Generate a scene-based evaluation dataset from held-out movies.

    For each non-training movie in *corpus*, picks the top
    *scenes_per_movie* longest scenes and generates a scene-summary query
    for each using the same prompt/logic as training (Type B queries).

    This directly aligns evaluation with the scene-level training
    objective and guarantees that every target movie has scenes in the
    corpus.

    Args:
        train_movie_ids: Movie IDs used for training (excluded).
        corpus: Full scene corpus (``build_scene_corpus(max_movies=None)``).
        output_path: Where to write the JSON.  Defaults to
            ``data/finetuning/eval_queries.json``.
        scenes_per_movie: How many scenes per movie to generate queries
            for.  Defaults to ``TOP_SCENES_FOR_SUMMARY`` (3).
        api_key: OpenRouter API key override.
        seed: RNG seed for shuffling candidate scenes.

    Returns:
        The path to the saved JSON file.
    """
    output = output_path or (FINETUNING_DATA_DIR / "eval_queries.json")
    qgen = QueryGenerator(api_key=api_key)
    rng = random.Random(seed)

    eval_movies = {
        mid: entry for mid, entry in corpus.items()
        if mid not in train_movie_ids
    }
    logger.info(
        "Eval pool: %d movies (corpus=%d, training=%d)",
        len(eval_movies), len(corpus), len(train_movie_ids),
    )

    # Check for existing partial output so we can resume
    existing: List[Dict[str, str]] = []
    done_keys: Set[str] = set()
    if output.exists():
        try:
            with open(output, "r", encoding="utf-8") as f:
                existing = json.load(f)
            done_keys = {f"{q['movie_id']}::{q.get('_scene_idx', '')}" for q in existing}
            logger.info("Resuming: %d queries already generated", len(existing))
        except (json.JSONDecodeError, KeyError):
            existing = []

    eval_queries: List[Dict[str, str]] = list(existing)
    movie_list = sorted(eval_movies.items(), key=lambda kv: kv[0])
    rng.shuffle(movie_list)

    total_movies = len(movie_list)
    generated = 0
    skipped_scenes = 0

    for movie_idx, (mid, entry) in enumerate(movie_list, 1):
        sorted_scenes = sorted(entry.scenes, key=lambda s: len(s.text), reverse=True)
        top_scenes = sorted_scenes[:scenes_per_movie]

        for scene_rank, scene in enumerate(top_scenes):
            resume_key = f"{mid}::{scene_rank}"
            if resume_key in done_keys:
                continue

            summary = qgen.generate_scene_summary(scene.text, entry.movie_title)
            if summary is None:
                skipped_scenes += 1
                continue

            eval_queries.append({
                "query": summary,
                "movie_id": mid,
                "movie_title": entry.movie_title,
                "_scene_idx": scene_rank,
            })
            generated += 1

        if movie_idx % 10 == 0 or movie_idx == total_movies:
            print(
                f"  [{movie_idx}/{total_movies}] "
                f"{generated} new queries generated, {skipped_scenes} skipped"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(eval_queries, f, ensure_ascii=False, indent=2)

    # Final write
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(eval_queries, f, ensure_ascii=False, indent=2)

    logger.info(
        "Scene-eval dataset complete: %d queries from %d movies → %s",
        len(eval_queries), total_movies, output,
    )
    print(
        f"\n  Eval dataset: {len(eval_queries)} queries "
        f"({generated} new, {len(existing)} resumed, {skipped_scenes} scenes skipped)"
    )
    return output
