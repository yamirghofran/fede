"""Generate a scene-based evaluation dataset from held-out movies.

Usage:
    python -m finetuning.scripts.generate_eval [--scenes-per-movie 3] [--seed 42]

Backs up the existing eval_queries.json before overwriting.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from finetuning.config import FINETUNING_DATA_DIR, TOP_SCENES_FOR_SUMMARY
from finetuning.corpus.scene_corpus import build_scene_corpus
from finetuning.evaluation.dataset_generator import generate_scene_eval_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scene-based eval dataset")
    parser.add_argument(
        "--scenes-per-movie", type=int, default=TOP_SCENES_FOR_SUMMARY,
        help=f"Scenes to query per eval movie (default: {TOP_SCENES_FOR_SUMMARY})",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-movies", type=int, default=1200)
    args = parser.parse_args()

    log.info("Building full scene corpus …")
    full_corpus = build_scene_corpus(max_movies=None, seed=args.seed)

    log.info("Building training corpus (%d movies) …", args.train_movies)
    train_corpus = build_scene_corpus(max_movies=args.train_movies, seed=args.seed)
    train_ids = set(train_corpus.keys())

    eval_pool = {mid for mid in full_corpus if mid not in train_ids}
    log.info(
        "Full corpus: %d | Training: %d | Eval pool: %d",
        len(full_corpus), len(train_ids), len(eval_pool),
    )

    output_path = FINETUNING_DATA_DIR / "eval_queries.json"
    backup_path = FINETUNING_DATA_DIR / "eval_queries.old.json"
    if output_path.exists():
        shutil.copy2(output_path, backup_path)
        log.info("Backed up existing eval queries → %s", backup_path.name)
        output_path.unlink()

    result = generate_scene_eval_dataset(
        train_movie_ids=train_ids,
        corpus=full_corpus,
        output_path=output_path,
        scenes_per_movie=args.scenes_per_movie,
        seed=args.seed,
    )

    with open(result, "r", encoding="utf-8") as f:
        queries = json.load(f)

    unique_movies = {q["movie_id"] for q in queries}
    print(f"\nDone: {len(queries)} queries from {len(unique_movies)} movies → {result}")


if __name__ == "__main__":
    main()
