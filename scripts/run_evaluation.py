"""
Run evaluation using the pipeline with a given retriever.
Currently wired to BM25Retriever as the baseline.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from evaluation.baselines.bm25_baseline import BM25Retriever
from evaluation.metrics.metrics import save_metrics_report
from evaluation.pipeline.evaluation_pipeline import run_pipeline

REPORT_PATH = os.path.join(project_root, "evaluation", "evaluation_dataset", "evaluation_report.json")


def print_results(metrics: dict):
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nTotal queries: {metrics['total_queries']}")

    print("\nAccuracy@k:")
    for k in metrics["metadata"]["k_values"]:
        acc = metrics["summary"][f"accuracy_at_{k}"]
        print(f"   Accuracy@{k}: {acc:.3f}  ({acc * 100:.1f}%)")

    print(f"\nMRR: {metrics['summary']['mrr']:.3f}")
    print(f"Median rank: {metrics['summary']['median_rank']}")
    print(f"Failed:{metrics['failed_queries']['count']} ({metrics['failed_queries']['count'] / metrics['total_queries'] * 100:.1f}%)")

    if metrics["failed_queries"]["count"] > 0:
        print("\nSample failed queries (first 3):")
        for q in metrics["failed_queries"]["queries"][:3]:
            print(f"[{q['query_id']}] {q['movie_name']}")
            print(f"{q['query'][:80]}...")


def main():
    print("=" * 70)
    print("FEDE — EVALUATION")
    print("=" * 70)

    print("\nInitializing BM25 retriever...")
    retriever = BM25Retriever()
    print(f"Index built: {len(retriever.movie_keys)} movies")

    print("\nRunning pipeline...")
    metrics = run_pipeline(retriever)

    print_results(metrics)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    save_metrics_report(metrics, REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
