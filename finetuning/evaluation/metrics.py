"""Retrieval evaluation metrics for the FEDE finetuning pipeline.

Provides Accuracy@k and Mean Reciprocal Rank (MRR) at the movie level.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


def accuracy_at_k(
    retrieved: List[Dict[str, Any]],
    target_movie_id: str,
    k: int,
) -> float:
    """Return 1.0 if *target_movie_id* appears in the top-*k* results, else 0.0."""
    for result in retrieved[:k]:
        if result.get("movie_id") == target_movie_id:
            return 1.0
    return 0.0


def mean_reciprocal_rank(
    retrieved: List[Dict[str, Any]],
    target_movie_id: str,
) -> float:
    """Return 1/rank if *target_movie_id* is found, else 0.0."""
    for i, result in enumerate(retrieved, start=1):
        if result.get("movie_id") == target_movie_id:
            return 1.0 / i
    return 0.0


def evaluate_batch(
    queries: List[Dict[str, Any]],
    retrieval_results: List[List[Dict[str, Any]]],
    k_values: List[int] | None = None,
) -> Dict[str, Any]:
    """Evaluate a batch of queries and aggregate metrics.

    Args:
        queries: Each dict must contain ``movie_id`` (the ground-truth
            movie) and ``query`` (for diagnostics).
        retrieval_results: One result list per query.  Each result dict
            must contain ``movie_id``.
        k_values: Values of k for Accuracy@k.  Defaults to ``[5, 10, 20]``.

    Returns:
        A dict with ``summary`` (aggregated scores), ``failed_queries``
        (queries not found in top max-k), and ``metadata``.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    total = len(queries)
    correct_counts = {k: 0 for k in k_values}
    mrr_sum = 0.0
    ranks: List[int] = []
    failed: List[Dict[str, Any]] = []

    for query, results in zip(queries, retrieval_results):
        target = query["movie_id"]

        rank = None
        for i, r in enumerate(results, start=1):
            if r.get("movie_id") == target:
                rank = i
                ranks.append(i)
                break

        for k in k_values:
            if rank is not None and rank <= k:
                correct_counts[k] += 1

        mrr_sum += mean_reciprocal_rank(results, target)

        max_k = max(k_values)
        if rank is None or rank > max_k:
            failed.append({
                "query": query.get("query", ""),
                "movie_id": target,
                "movie_title": query.get("movie_title", ""),
                "rank": rank,
            })

    accuracies = {f"accuracy_at_{k}": correct_counts[k] / total for k in k_values}
    mrr = mrr_sum / total if total else 0.0
    median_rank = sorted(ranks)[len(ranks) // 2] if ranks else None

    return {
        "total_queries": total,
        "summary": {
            **{f"correct_at_{k}": correct_counts[k] for k in k_values},
            **accuracies,
            "mrr": round(mrr, 4),
            "median_rank": median_rank,
        },
        "failed_queries": {
            "count": len(failed),
            "queries": failed,
        },
        "metadata": {
            "k_values": k_values,
            "metrics": ["Accuracy@k", "MRR"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
