"""
1. Loading the generated queries
2. Running retrieval using BM25 baseline
3. Evaluating results with the metrics module
4. Saving the evaluation report
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json

from evaluation.baselines.bm25_baseline import BM25Retriever
from evaluation.metrics import evaluate_batch, save_evaluation_report


def main():
    print("=" * 70)
    print("MOVIE SEARCH ENGINE EVALUATION")
    print("=" * 70)

    # 1. Load generated queries
    print("\n1. Loading queries...")
    queries_path = os.path.join(
        project_root, "evaluation_dataset", "generated_queries.json"
    )

    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        queries = data["evaluation_queries"]

    print(f"   Loaded {len(queries)} queries")

    # 2. Initialize retriever
    print("\n2. Initializing BM25 retriever...")
    retriever = BM25Retriever()
    print(f"   Index built with {len(retriever.movie_keys)} movies")

    # 3. Run retrieval for all queries
    print("\n3. Running retrieval for all queries...")
    retrieval_results = []

    for i, query in enumerate(queries, start=1):
        # Retrieve top 20 results (needed for k=20 evaluation)
        results = retriever.retrieve(query["query"], top_k=20)
        retrieval_results.append(results)

        if i % 50 == 0:
            print(f"   Processed {i}/{len(queries)} queries...")

    print(f"   Completed {len(queries)} queries")

    # 4. Evaluate results
    print("\n4. Evaluating results...")
    metrics = evaluate_batch(queries, retrieval_results)

    # 5. Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal queries: {metrics['total_queries']}")

    print("\nCorrect counts:")
    for k in metrics["metadata"]["k_values"]:
        print(f"   Correct at {k}: {metrics['summary'][f'correct_at_{k}']}")

    print("\nAccuracy at k:")
    for k in metrics["metadata"]["k_values"]:
        acc = metrics["summary"][f"accuracy_at_{k}"]
        print(f"   Accuracy@{k}: {acc:.3f} ({acc * 100:.1f}%)")

    print(f"\nMean Reciprocal Rank: {metrics['summary']['mrr']:.3f}")
    print(f"Median Rank: {metrics['summary']['median_rank']}")

    print(
        f"\nFailed queries: {metrics['failed_queries']['count']} ({metrics['failed_queries']['count'] / metrics['total_queries'] * 100:.1f}%)"
    )

    # Show first few failed queries
    if metrics["failed_queries"]["count"] > 0:
        print("\nSample failed queries (first 3):")
        for q in metrics["failed_queries"]["queries"][:3]:
            print(f"   Query {q['query_id']}: {q['movie_name']}")
            print(f"   Query text: {q['query'][:80]}...")
            print()

    # 6. Save evaluation report
    print("=" * 70)
    report_path = os.path.join(
        project_root, "evaluation_dataset", "evaluation_report.json"
    )
    save_evaluation_report(metrics, report_path)
    print(f"\nEvaluation report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
