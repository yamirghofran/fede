#!/usr/bin/env python3
"""CLI: Train (or continue training) the FEDE embedding model.

Usage::

    # Round 1 — from base model, with per-epoch evaluation
    python -m finetuning.scripts.train \\
        --round 1 \\
        --output fede-embeddinggemma/round1 \\
        --eval-dataset data/finetuning/eval_queries.json

    # Round 2 — from round-1 checkpoint with hard negatives
    python -m finetuning.scripts.train \\
        --round 2 \\
        --model fede-embeddinggemma/round1 \\
        --dataset data/finetuning/training_pairs_r2.jsonl \\
        --output fede-embeddinggemma/round2 \\
        --eval-dataset data/finetuning/eval_queries.json

    # Without evaluation (faster, no corpus encoding overhead)
    python -m finetuning.scripts.train --round 1 --output fede-embeddinggemma/round1 --no-eval

    # With LoRA
    python -m finetuning.scripts.train --round 1 --lora --output ...
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from finetuning.config import FINETUNING_DATA_DIR
from finetuning.training.model import load_model
from finetuning.training.trainer import build_evaluator, build_trainer, load_training_dataset

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the FEDE embedding model")
    parser.add_argument("--round", type=int, choices=[1, 2], required=True, help="Training round")
    parser.add_argument("--model", type=str, default=None, help="Model ID or checkpoint path (default: base model)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to training JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--eval-dataset", type=str, default=None, help="Path to eval_queries.json for per-epoch eval")
    parser.add_argument("--no-eval", action="store_true", help="Skip per-epoch evaluation")
    parser.add_argument("--eval-movies", type=int, default=None, help="Max movies in eval corpus (default: all)")
    parser.add_argument("--lora", action="store_true", help="Use LoRA adapters")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    default_dataset = f"training_pairs_r{args.round}.jsonl"
    dataset_path = Path(args.dataset) if args.dataset else (FINETUNING_DATA_DIR / default_dataset)

    model = load_model(args.model)
    train_dataset = load_training_dataset(dataset_path)

    # Build evaluator if eval dataset is available and not explicitly disabled
    evaluator = None
    if not args.no_eval:
        eval_path = Path(args.eval_dataset) if args.eval_dataset else (FINETUNING_DATA_DIR / "eval_queries.json")
        if eval_path.exists():
            from finetuning.corpus.scene_corpus import build_scene_corpus

            logger.info("Building eval corpus for per-epoch evaluation …")
            eval_corpus = build_scene_corpus(max_movies=args.eval_movies)
            evaluator = build_evaluator(eval_path, eval_corpus)
        else:
            logger.warning("Eval dataset not found at %s — training without per-epoch evaluation", eval_path)

    kwargs = {}
    if args.epochs is not None:
        kwargs["num_epochs"] = args.epochs
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr

    trainer = build_trainer(
        model=model,
        train_dataset=train_dataset,
        output_dir=args.output,
        evaluator=evaluator,
        use_lora=args.lora,
        **kwargs,
    )

    logger.info("Starting round-%d training …", args.round)
    trainer.train()

    model.save(args.output)
    logger.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
