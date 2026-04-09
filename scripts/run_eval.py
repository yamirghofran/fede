"""
Accuracy@k evaluation CLI.

Usage:
    python scripts/run_eval.py --methods bm25
    python scripts/run_eval.py --methods bm25 semantic hybrid
    python scripts/run_eval.py --methods hybrid --api-url http://localhost:8000

Methods: bm25, semantic, hybrid
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

UTC = timezone.utc

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from evaluation.pipeline.evaluation_pipeline import run_pipeline

# Whether each method is deterministic (affects default n_runs)
IS_DETERMINISTIC = {
    "bm25": True,
    "semantic": True,
    "hybrid": True,  # LLM uses temperature=0 → deterministic
}

DEFAULT_N_RUNS_NONDETERMINISTIC = 5
RESULTS_DIR = os.path.join(project_root, "evaluation", "results")


def _build_retriever(method: str, api_url: str):
    if method == "bm25":
        from evaluation.baselines.bm25_baseline import BM25Retriever
        r = BM25Retriever()
        print(f"  BM25 index: {len(r.movie_keys)} movies")
        return r
    else:
        from evaluation.baselines.api_retriever import ApiRetriever
        r = ApiRetriever(mode=method, base_url=api_url)
        if not r.health_check():
            print(f"  WARNING: API at {api_url} is not reachable. Results may be empty.")
        return r


def _resolve_n_runs(method: str, cli_n_runs: int | None) -> int:
    if cli_n_runs is not None:
        return cli_n_runs
    return 1 if IS_DETERMINISTIC[method] else DEFAULT_N_RUNS_NONDETERMINISTIC


def _print_result(method: str, n_runs: int, metrics: dict, k_values: list):
    summary = metrics["summary"]
    total = metrics["total_queries"]
    print(f"\n  {'─'*50}")
    print(f"  Method : {method}  (n_runs={n_runs})")
    print(f"  Queries: {total}")
    for k in k_values:
        acc = summary[f"accuracy_at_{k}"]
        print(f"    Accuracy@{k:<2}: {acc:.4f}  ({acc*100:.1f}%)")
    print(f"    MRR       : {summary['mrr']:.4f}")
    print(f"    Median rank: {summary['median_rank']}")
    failed = metrics["failed_queries"]["count"]
    print(f"    Failed    : {failed}/{total} ({failed/total*100:.1f}%)")


def _save_csv(rows: list, output_path: str, k_values: list):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = (
        ["method", "n_runs", "total_queries"]
        + [f"accuracy@{k}" for k in k_values]
        + ["mrr", "median_rank", "failed_queries", "evaluated_at"]
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Accuracy@k evaluation for movie retrieval methods")
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=["bm25", "semantic", "hybrid"],
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Runs per method. Defaults: 1 (deterministic), 5 (non-deterministic). "
             "Non-deterministic: hybrid.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        metavar="K",
        help="k cutoffs for Accuracy@k (default: 5 10 15 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: evaluation/results/accuracy_<timestamp>.csv",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the FEDE API (for semantic/hybrid)",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Exclude queries flagged for lexical leakage (287 queries instead of 298)",
    )
    args = parser.parse_args()

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    output_path = args.output or os.path.join(RESULTS_DIR, f"accuracy_{ts}.csv")

    print("=" * 60)
    print("FEDE — ACCURACY@k EVALUATION")
    print("=" * 60)
    print(f"Methods   : {', '.join(args.methods)}")
    print(f"k values  : {args.k_values}")
    print(f"Clean only: {args.clean_only} {'(287 queries)' if args.clean_only else '(298 queries, includes 11 leaky)'}")
    print(f"Output    : {output_path}")

    csv_rows = []

    for method in args.methods:
        n_runs = _resolve_n_runs(method, args.n_runs)
        print(f"\n[{method.upper()}] Initializing (n_runs={n_runs})...")

        retriever = _build_retriever(method, args.api_url)

        print(f"[{method.upper()}] Running pipeline...")
        metrics = run_pipeline(retriever, k_values=args.k_values, n_runs=n_runs, clean_only=args.clean_only)

        _print_result(method, n_runs, metrics, args.k_values)

        summary = metrics["summary"]
        csv_rows.append({
            "method": method,
            "n_runs": n_runs,
            "total_queries": metrics["total_queries"],
            **{f"accuracy@{k}": round(summary[f"accuracy_at_{k}"], 4) for k in args.k_values},
            "mrr": summary["mrr"],
            "median_rank": summary["median_rank"],
            "failed_queries": metrics["failed_queries"]["count"],
            "evaluated_at": ts,
        })

    _save_csv(csv_rows, output_path, args.k_values)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
