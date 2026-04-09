"""Evaluation pipeline — run any retriever against the held-out eval set.

A *retriever* is any object with a ``retrieve(query: str, top_k: int)``
method that returns ``List[Dict]`` where each dict contains at least
``movie_id``.  This keeps the pipeline decoupled from Qdrant specifics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from finetuning.config import EVAL_K_VALUES, FINETUNING_DATA_DIR
from finetuning.evaluation.metrics import evaluate_batch

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    """Minimal interface that any retriever must satisfy."""

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]: ...


def _load_eval_queries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if not queries:
        raise ValueError(f"Eval dataset is empty: {path}")
    return queries


def run_pipeline(
    retriever: Retriever,
    eval_path: Optional[Path] = None,
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run *retriever* against every eval query and return aggregated metrics.

    Args:
        retriever: Any object implementing the ``Retriever`` protocol.
        eval_path: Path to ``eval_queries.json``.  Defaults to
            ``data/finetuning/eval_queries.json``.
        k_values: Accuracy@k values.  Defaults to ``EVAL_K_VALUES``.

    Returns:
        The output of ``evaluate_batch`` — a dict with ``summary``,
        ``failed_queries``, and ``metadata``.
    """
    eval_path = eval_path or (FINETUNING_DATA_DIR / "eval_queries.json")
    k_values = k_values or list(EVAL_K_VALUES)
    top_k = max(k_values)

    queries = _load_eval_queries(eval_path)
    logger.info("Running evaluation: %d queries, top_k=%d", len(queries), top_k)

    retrieval_results: List[List[Dict[str, Any]]] = []
    for i, q in enumerate(queries, 1):
        results = retriever.retrieve(q["query"], top_k=top_k)
        retrieval_results.append(results)
        if i % 25 == 0:
            logger.info("  evaluated %d / %d queries", i, len(queries))

    metrics = evaluate_batch(queries, retrieval_results, k_values)
    logger.info(
        "Evaluation complete — MRR=%.4f, Acc@%d=%.3f",
        metrics["summary"]["mrr"],
        k_values[0],
        metrics["summary"][f"accuracy_at_{k_values[0]}"],
    )
    return metrics
