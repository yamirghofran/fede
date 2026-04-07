import json
import os
from typing import Dict, List

from evaluation.metrics.metrics import evaluate_batch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "evaluation", "evaluation_dataset", "generated_queries.json")


def _load_dataset(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("evaluation_queries", [])
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def run_pipeline(retriever, dataset_path: str = DEFAULT_DATASET_PATH, k_values: List[int] = [5, 10, 15, 20]) -> Dict:
    """Run a retriever against the eval dataset and return aggregated metrics."""
    queries = _load_dataset(dataset_path)
    top_k = max(k_values)

    retrieval_results = []
    for q in queries:
        retrieval_results.append(retriever.retrieve(q["query"], top_k=top_k))

    return evaluate_batch(queries, retrieval_results, k_values)
