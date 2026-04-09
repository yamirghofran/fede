"""
Retrieval Evaluation Metrics for Movie Search Engine
Following Mustafa et al. [4] - IJECE Vol. 14, No. 6
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
import math


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


def ndcg_at_k(retrieved: List[Dict], relevance_judgments: Dict[str, float], k: int) -> float:
    """
    Calculate NDCG@k for a single query.

    Args:
        retrieved: Ordered list of results with 'movie_key' field
        relevance_judgments: {movie_key: grade} where grade ∈ {0.0, 1.0, 2.0}
        k: cutoff

    Returns:
        NDCG@k score (0.0 if no relevant documents in judgments)
    """
    top_k = retrieved[:k]

    # DCG: sum of rel_i / log2(i+1) for i=1..k
    dcg = 0.0
    for i, result in enumerate(top_k, start=1):
        rel = relevance_judgments.get(result["movie_key"], 0.0)
        dcg += rel / math.log2(i + 1)

    # IDCG: ideal ranking (sort grades descending)
    ideal_grades = sorted(relevance_judgments.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_grades, start=1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_batch_ndcg(
    queries: List[Dict],
    retrieval_results: List[List[Dict]],
    relevance_judgments_per_query: Dict[str, Dict[str, float]],
    k: int = 5,
) -> Dict:
    """
    Compute mean NDCG@k over a batch.

    Args:
        queries: List of query dicts with 'id' field
        retrieval_results: Retrieval results per query (same order as queries)
        relevance_judgments_per_query: {query_id: {movie_key: grade}}
        k: cutoff

    Returns:
        Dict with mean_ndcg, per_query scores, and metadata
    """
    scores = []
    per_query = []

    for query, results in zip(queries, retrieval_results):
        qid = str(query["id"])
        judgments = relevance_judgments_per_query.get(qid, {})
        score = ndcg_at_k(results, judgments, k)
        scores.append(score)
        per_query.append({"query_id": qid, "movie_key": query["movie_key"], f"ndcg@{k}": round(score, 4)})

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
        f"mean_ndcg@{k}": round(mean_score, 4),
        "n_queries": len(queries),
        "per_query": per_query,
    }


def save_metrics_report(metrics: Dict, output_path: str) -> None:
    """Save metrics to evaluation_report.json"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
