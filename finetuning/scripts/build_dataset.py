#!/usr/bin/env python3
"""CLI: Build the round-1 training dataset.

Usage::

    python -m finetuning.scripts.build_dataset --movies 1000
    python -m finetuning.scripts.build_dataset --movies 500 --output my_pairs.jsonl --no-resume
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from finetuning.config import FINETUNING_DATA_DIR
from finetuning.dataset.dataset_builder import DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build round-1 finetuning dataset")
    parser.add_argument("--movies", type=int, default=None, help="Max movies to process (default: all)")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output = Path(args.output) if args.output else (FINETUNING_DATA_DIR / "training_pairs_r1.jsonl")

    builder = DatasetBuilder(
        max_movies=args.movies,
        api_key=args.api_key,
    )
    builder.build(output_path=output, resume=not args.no_resume)


if __name__ == "__main__":
    main()
