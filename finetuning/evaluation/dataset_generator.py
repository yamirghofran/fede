"""Generate a held-out evaluation query set for movie-level retrieval.

Produces ~``EVAL_DATASET_SIZE`` queries from movies that were *not* used
in training.  Each query is a 1-sentence whole-movie description generated
by the LLM, paired with the ground-truth ``movie_id`` so that
``evaluate_batch`` can check retrieval correctness.

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

from finetuning.config import EVAL_DATASET_SIZE, FINETUNING_DATA_DIR
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

    Returns:
        The path to the saved JSON file.
    """
    output = output_path or (FINETUNING_DATA_DIR / "eval_queries.json")
    qgen = QueryGenerator(api_key=api_key)

    # Build candidate pool: movies with a usable overview that are NOT in the training set
    candidates: List[Dict[str, Any]] = []
    for key, entry in metadata.items():
        file_info = entry.get("file", {})
        tmdb_info = entry.get("tmdb", {})
        movie_title = file_info.get("name", key)
        movie_id = movie_title.lower().replace(" ", "_").replace("-", "_")

        if movie_id in corpus_movie_ids:
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
