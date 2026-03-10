import json
import os
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "evaluation_dataset", "generated_queries.json")


def _load_dataset(path: str) -> List[Dict]:
    """Load and validate the eval dataset. Returns the list of query entries."""
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval dataset not found at {path}\n")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data.get("evaluation_queries", [])
    
    if not queries:
        raise ValueError(f"No queries found in {path}")

    return queries


def _find_correct_rank(results: List[Dict], target_key: str) -> Optional[int]:
    """Return the 1-based rank of target_key in results, or None if not found."""

    for r in results:
        if r["movie_key"] == target_key:
            return r["rank"]
    return None


def _run_retrieval_loop(retriever, queries: List[Dict], top_k: int) -> List[Dict]:
    """Run every query through the retriever. Returns a list of RunResults."""

    per_query = []

    for q in queries:
        results = retriever.retrieve(q["query"], top_k=top_k)
        correct_rank = _find_correct_rank(results, q["movie_key"])

        per_query.append({
            "query_id": q["id"],
            "query": q["query"],
            "movie_key": q["movie_key"],
            "movie_name": q["movie_name"],
            "results": results,
            "correct_rank": correct_rank,
        })

    found = sum(1 for r in per_query if r["correct_rank"] is not None)
    print(f"Found correct movie in top {top_k} for {found}/{len(per_query)} queries ({found/len(per_query):.2%})")

    return per_query


def _compute_metrics(per_query: List[Dict]) -> Dict:
    return None  # placeholder until metrics are implemented


def run_pipeline(retriever, dataset_path: str = DEFAULT_DATASET_PATH, top_k: int = 20,) -> Dict:
    """Run a retriever against the eval dataset and return a PipelineResult."""

    queries = _load_dataset(dataset_path)
    per_query = _run_retrieval_loop(retriever, queries, top_k)
    metrics = _compute_metrics(per_query)

    return {
        "retriever":   retriever.name,
        "num_queries": len(queries),
        "top_k":       top_k,
        "per_query":   per_query,
        "metrics":     metrics,
    }
