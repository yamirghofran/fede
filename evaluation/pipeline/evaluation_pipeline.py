import json
import os
from typing import Dict, List, Optional

from tqdm import tqdm

from evaluation.metrics.metrics import evaluate_batch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "evaluation", "evaluation_dataset", "generated_queries.json")


def _load_dataset(path: str, clean_only: bool = False) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("evaluation_queries", [])
    if not queries:
        raise ValueError(f"No queries found in {path}")
    if clean_only:
        queries = [q for q in queries if not q.get("validation", {}).get("has_leakage", False)]
    return queries


def _rrf_merge(all_run_results: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.
    Returns a single merged ranking sorted by descending RRF score.
    """
    rrf_scores: Dict[str, float] = {}
    movie_meta: Dict[str, Dict] = {}

    for run_results in all_run_results:
        for rank, result in enumerate(run_results, start=1):
            mkey = result["movie_key"]
            rrf_scores[mkey] = rrf_scores.get(mkey, 0.0) + 1.0 / (k + rank)
            if mkey not in movie_meta:
                movie_meta[mkey] = result

    merged = sorted(rrf_scores.keys(), key=lambda m: rrf_scores[m], reverse=True)
    return [{**movie_meta[m], "score": rrf_scores[m]} for m in merged]


def run_pipeline(
    retriever,
    dataset_path: str = DEFAULT_DATASET_PATH,
    k_values: List[int] = [5, 10, 15, 20],
    n_runs: int = 1,
    clean_only: bool = False,
) -> Dict:
    """
    Run a retriever against the eval dataset and return aggregated metrics.

    Args:
        retriever: Object with .retrieve(query, top_k) -> List[Dict] interface
        dataset_path: Path to generated_queries.json
        k_values: List of k cutoffs for Accuracy@k
        n_runs: Number of runs. >1 triggers RRF aggregation (for non-deterministic retrievers)
        clean_only: If True, exclude queries flagged for lexical leakage
    """
    queries = _load_dataset(dataset_path, clean_only=clean_only)
    top_k = max(k_values)

    retrieval_results = []
    for q in tqdm(queries, desc="Retrieving", unit="query"):
        if n_runs == 1:
            results = retriever.retrieve(q["query"], top_k=top_k)
        else:
            # Multiple runs — aggregate with RRF
            run_lists = [retriever.retrieve(q["query"], top_k=top_k) for _ in range(n_runs)]
            results = _rrf_merge(run_lists)

        retrieval_results.append(results)

    return evaluate_batch(queries, retrieval_results, k_values)
