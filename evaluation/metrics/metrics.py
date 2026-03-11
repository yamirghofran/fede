"""
Retrieval Evaluation Metrics for Movie Search Engine
Following Mustafa et al. [4] - IJECE Vol. 14, No. 6
"""

from typing import Dict, List
from datetime import datetime
import json


def accuracy_at_k(retrieved: List[Dict], target_movie_key: str, k: int) -> float:
    """
    Calculate Accuracy@k for a single query.

    Returns 1.0 if target movie is in top-k, 0.0 otherwise.
    """
    top_k = retrieved[:k]
    movie_keys = [r["movie_key"] for r in top_k]
    return 1.0 if target_movie_key in movie_keys else 0.0


def mean_reciprocal_rank(retrieved: List[Dict], target_movie_key: str) -> float:
    """
    Calculate MRR for a single query.

    Returns 1/rank if target movie is found, 0.0 otherwise.
    Higher is better (1.0 = movie is #1).
    """
    for i, result in enumerate(retrieved, start=1):
        if result["movie_key"] == target_movie_key:
            return 1.0 / i
    return 0.0


def evaluate_batch(
    queries: List[Dict],
    retrieval_results: List[List[Dict]],
    k_values: List[int] = [5, 10, 15, 20],
) -> Dict:
    """
    Evaluate batch of queries and aggregate metrics.

    Args:
        queries: List of query dicts with 'id', 'query', 'movie_name', 'movie_key'
        retrieval_results: List of retrieval results for each query
        k_values: List of k values for Accuracy@k

    Returns:
        Aggregated metrics dict with summary and failed queries
    """
    total_queries = len(queries)

    # Initialize counters
    correct_counts = {k: 0 for k in k_values}
    mrr_sum = 0.0

    # Track failed queries and ranks for analysis
    failed_queries = []
    ranks = []

    for query, results in zip(queries, retrieval_results):
        target_key = query["movie_key"]

        # Find rank of correct movie
        rank = None
        for i, result in enumerate(results, start=1):
            if result["movie_key"] == target_key:
                rank = i
                ranks.append(i)
                break

        # Calculate accuracy at each k
        for k in k_values:
            if rank and rank <= k:
                correct_counts[k] += 1

        # Calculate MRR
        mrr_sum += mean_reciprocal_rank(results, target_key)

        # Track failed queries (not found in top 20)
        if rank is None or rank > 20:
            failed_queries.append(
                {
                    "query_id": query["id"],
                    "query": query["query"],
                    "movie_name": query["movie_name"],
                    "movie_key": target_key,
                    "rank": rank,
                }
            )

    # Calculate averages
    avg_accuracies = {
        f"accuracy_at_{k}": correct_counts[k] / total_queries for k in k_values
    }
    avg_mrr = mrr_sum / total_queries

    # Calculate median rank
    median_rank = sorted(ranks)[len(ranks) // 2] if ranks else None

    return {
        "total_queries": total_queries,
        "summary": {
            **{f"correct_at_{k}": correct_counts[k] for k in k_values},
            **avg_accuracies,
            "mrr": round(avg_mrr, 3),
            "median_rank": median_rank,
        },
        "failed_queries": {"count": len(failed_queries), "queries": failed_queries},
        "metadata": {
            "k_values": k_values,
            "metrics": ["Accuracy@k", "MRR"],
            "methodology": "Mustafa et al. [4] - IJECE Vol. 14, No. 6",
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
        },
    }


def save_evaluation_report(metrics: Dict, output_path: str) -> None:
    """Save metrics to evaluation_report.json"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
