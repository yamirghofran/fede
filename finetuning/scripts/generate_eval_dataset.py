#!/usr/bin/env python3
"""CLI: Generate the held-out evaluation query set.

Reads the training checkpoint to find which movies were already used in
training, then generates ~100 evaluation queries from the *remaining*
movies (those that exist in the metadata but were NOT included in the
training corpus).

Run this AFTER build_dataset and BEFORE train.

Usage::

    python -m finetuning.scripts.generate_eval_dataset
    python -m finetuning.scripts.generate_eval_dataset --n 150 --output my_eval.json
    python -m finetuning.scripts.generate_eval_dataset --training-movies 900
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Set

from finetuning.config import EVAL_DATASET_SIZE, FINETUNING_DATA_DIR
from finetuning.corpus.scene_corpus import load_metadata
from finetuning.dataset.query_generator import load_checkpoint
from finetuning.evaluation.dataset_generator import generate_eval_dataset

logger = logging.getLogger(__name__)


def _training_movie_ids_from_checkpoint() -> Set[str]:
    """Read processed movie IDs from the dataset build checkpoint."""
    ckpt = load_checkpoint()
    if ckpt:
        ids = set(ckpt.get("processed_movies", []))
        logger.info("Loaded %d training movie IDs from checkpoint", len(ids))
        return ids
    logger.warning("No checkpoint found — eval pool will be drawn from ALL metadata movies")
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate held-out evaluation queries (run after build_dataset)"
    )
    parser.add_argument(
        "--n", type=int, default=EVAL_DATASET_SIZE,
        help=f"Number of eval queries to generate (default: {EVAL_DATASET_SIZE})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: data/finetuning/eval_queries.json)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    training_ids = _training_movie_ids_from_checkpoint()

    logger.info("Loading metadata …")
    metadata = load_metadata()
    logger.info("Metadata loaded: %d entries", len(metadata))

    output = Path(args.output) if args.output else (FINETUNING_DATA_DIR / "eval_queries.json")

    generate_eval_dataset(
        corpus_movie_ids=training_ids,
        metadata=metadata,
        output_path=output,
        n=args.n,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
