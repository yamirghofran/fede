#!/usr/bin/env python3
"""CLI: Mine hard negatives using the round-1 model and produce the round-2 dataset.

Usage::

    python -m finetuning.scripts.mine_hard_negatives \\
        --model fede-embeddinggemma/round1 \\
        --dataset data/finetuning/training_pairs_r1.jsonl \\
        --output data/finetuning/training_pairs_r2.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from finetuning.config import FINETUNING_DATA_DIR, HARD_NEGATIVES_PER_QUERY
from finetuning.corpus.scene_corpus import build_scene_corpus
from finetuning.dataset.negative_miner import CorpusIndex, mine_hard_negatives
from finetuning.training.model import load_model

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives for round-2 training")
    parser.add_argument("--model", type=str, required=True, help="Path to round-1 model checkpoint")
    parser.add_argument("--dataset", type=str, default=None, help="Round-1 JSONL dataset path")
    parser.add_argument("--output", type=str, default=None, help="Round-2 JSONL output path")
    parser.add_argument("--movies", type=int, default=None, help="Max movies for corpus (default: all)")
    parser.add_argument("--hard-negs", type=int, default=HARD_NEGATIVES_PER_QUERY, help="Hard negatives per query")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    r1_path = Path(args.dataset) if args.dataset else (FINETUNING_DATA_DIR / "training_pairs_r1.jsonl")
    r2_path = Path(args.output) if args.output else (FINETUNING_DATA_DIR / "training_pairs_r2.jsonl")

    model = load_model(args.model)

    logger.info("Building scene corpus …")
    corpus = build_scene_corpus(max_movies=args.movies)

    logger.info("Building corpus embedding index …")
    index = CorpusIndex.build(corpus, model)

    logger.info("Loading round-1 dataset from %s …", r1_path)
    with open(r1_path, "r", encoding="utf-8") as f:
        r1_rows = [json.loads(line) for line in f]

    logger.info("Mining hard negatives for %d pairs …", len(r1_rows))
    r2_path.parent.mkdir(parents=True, exist_ok=True)
    with open(r2_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(r1_rows, 1):
            hard_negs = mine_hard_negatives(
                query=row["anchor"],
                movie_id=row["movie_id"],
                index=index,
                model=model,
                n=args.hard_negs,
            )
            row["negatives"] = hard_negs
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

            if i % 500 == 0:
                logger.info("  processed %d / %d", i, len(r1_rows))

    logger.info("Round-2 dataset written: %d pairs → %s", len(r1_rows), r2_path)


if __name__ == "__main__":
    main()
