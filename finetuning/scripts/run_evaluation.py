#!/usr/bin/env python3
"""CLI: Run the full evaluation pipeline and save the report.

Usage::

    python -m finetuning.scripts.run_evaluation --model fede-embeddinggemma/round2
    python -m finetuning.scripts.run_evaluation --model fede-embeddinggemma/round2 --eval-dataset data/finetuning/eval_queries.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from finetuning.config import FINETUNING_DATA_DIR
from finetuning.evaluation.pipeline import run_pipeline
from finetuning.evaluation.semantic_retriever import SemanticRetriever
from finetuning.training.model import load_model

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned embedding model")
    parser.add_argument("--model", type=str, required=True, help="Model ID or checkpoint path")
    parser.add_argument("--eval-dataset", type=str, default=None, help="Path to eval_queries.json")
    parser.add_argument("--report", type=str, default=None, help="Path to save eval report JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model = load_model(args.model)
    retriever = SemanticRetriever(model)

    eval_path = Path(args.eval_dataset) if args.eval_dataset else None
    metrics = run_pipeline(retriever, eval_path=eval_path)

    report_path = Path(args.report) if args.report else (FINETUNING_DATA_DIR / "eval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    summary = metrics["summary"]
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total queries:   {metrics['total_queries']}")
    print(f"  MRR:             {summary['mrr']:.4f}")
    for key, val in summary.items():
        if key.startswith("accuracy_at_"):
            k = key.split("_")[-1]
            correct = summary.get(f"correct_at_{k}", "?")
            print(f"  Accuracy@{k}:     {val:.3f}  ({correct}/{metrics['total_queries']})")
    print(f"  Median rank:     {summary.get('median_rank', 'N/A')}")
    print(f"  Failed (>{max(metrics['metadata']['k_values'])}): {metrics['failed_queries']['count']}")
    print(f"\n  Report saved to: {report_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
