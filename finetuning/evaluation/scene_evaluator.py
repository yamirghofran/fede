"""Scene-level max-pool evaluation — no Qdrant required.

Encodes every scene in the corpus individually with the given model,
then for each eval query computes ``max(cosine_sim)`` across all scenes
of each movie.  The movie with the highest max-scene similarity is the
top retrieval result.

This aligns evaluation with the scene-level training objective while
still reporting movie-level Accuracy@k and MRR.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from finetuning.config import EVAL_K_VALUES, FINETUNING_DATA_DIR
from finetuning.corpus.scene_corpus import MovieEntry
from finetuning.evaluation.metrics import evaluate_batch
from finetuning.training.model import encode_documents, encode_queries

logger = logging.getLogger(__name__)


class ScenePoolEvaluator:
    """Encodes all scenes once, then evaluates queries via max-pool per movie."""

    def __init__(
        self,
        model: SentenceTransformer,
        corpus: Dict[str, MovieEntry],
        batch_size: int = 64,
    ) -> None:
        self._model = model
        self._corpus = corpus

        scene_texts: List[str] = []
        movie_ids: List[str] = []
        for mid, entry in corpus.items():
            for scene in entry.scenes:
                scene_texts.append(scene.text)
                movie_ids.append(mid)

        print(f"  Encoding {len(scene_texts)} scenes from {len(corpus)} movies …")
        self._scene_embs = encode_documents(
            model, scene_texts, batch_size=batch_size, show_progress=True,
        )
        norms = np.linalg.norm(self._scene_embs[:5], axis=1)
        print(f"  Embedding shape: {self._scene_embs.shape}, dtype: {self._scene_embs.dtype}")
        print(f"  First 5 norms: {norms}")
        self._movie_ids = np.array(movie_ids)
        self._unique_movies = {mid: corpus[mid].movie_title for mid in corpus}

    @property
    def corpus_movie_ids(self) -> set:
        """Movie IDs present in the encoded corpus."""
        return set(self._unique_movies.keys())

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Return top-k movies ranked by max scene similarity."""
        q_emb = encode_queries(self._model, query)
        sims = self._scene_embs @ q_emb

        best_per_movie: Dict[str, float] = {}
        for idx in range(len(sims)):
            mid = str(self._movie_ids[idx])
            score = float(sims[idx])
            if mid not in best_per_movie or score > best_per_movie[mid]:
                best_per_movie[mid] = score

        ranked = sorted(best_per_movie.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {
                "movie_id": mid,
                "movie_title": self._unique_movies.get(mid, mid),
                "score": score,
            }
            for mid, score in ranked[:top_k]
        ]


def run_scene_eval(
    model: SentenceTransformer,
    corpus: Dict[str, MovieEntry],
    eval_path: Optional[Path] = None,
    k_values: Optional[List[int]] = None,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """One-call convenience: build a ``ScenePoolEvaluator`` and run it.

    Queries whose target ``movie_id`` is absent from *corpus* are
    automatically excluded so that missing movies don't inflate the
    failure count.  The number of skipped queries is reported in
    ``metadata.skipped_missing_target``.

    Returns the same dict shape as ``pipeline.run_pipeline`` — has
    ``summary``, ``failed_queries``, and ``metadata`` keys.
    """
    eval_path = eval_path or (FINETUNING_DATA_DIR / "eval_queries.json")
    k_values = k_values or list(EVAL_K_VALUES)
    top_k = max(k_values)

    with open(eval_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    evaluator = ScenePoolEvaluator(model, corpus, batch_size=batch_size)
    corpus_ids = evaluator.corpus_movie_ids

    valid_queries = [q for q in all_queries if q["movie_id"] in corpus_ids]
    skipped = len(all_queries) - len(valid_queries)
    if skipped:
        logger.warning(
            "%d / %d eval queries target movies NOT in the scene corpus — "
            "these will be excluded from metrics.",
            skipped, len(all_queries),
        )
    print(
        f"  Eval queries: {len(all_queries)} total, "
        f"{len(valid_queries)} valid (target in corpus), "
        f"{skipped} skipped"
    )

    if not valid_queries:
        logger.error("No valid eval queries — every target movie is missing from the corpus!")
        return {
            "total_queries": 0,
            "summary": {
                **{f"correct_at_{k}": 0 for k in k_values},
                **{f"accuracy_at_{k}": 0.0 for k in k_values},
                "mrr": 0.0,
                "median_rank": None,
            },
            "failed_queries": {"count": 0, "queries": []},
            "metadata": {
                "k_values": k_values,
                "skipped_missing_target": skipped,
            },
        }

    logger.info("Running scene-pool evaluation: %d valid queries, top_k=%d", len(valid_queries), top_k)
    retrieval_results: List[List[Dict[str, Any]]] = []
    for i, q in enumerate(valid_queries, 1):
        results = evaluator.retrieve(q["query"], top_k=top_k)
        retrieval_results.append(results)

        if i == 1:
            target = q["movie_id"]
            top3 = results[:3]
            target_in_results = any(r["movie_id"] == target for r in results)
            print(f"  Sample query: {q['query'][:80]}…")
            print(f"  Target: {target} | found in top-{top_k}: {target_in_results}")
            for j, r in enumerate(top3, 1):
                tag = " ◀ target" if r["movie_id"] == target else ""
                print(f"    #{j} {r['movie_id']:<35} score={r['score']:.4f}{tag}")

        if i % 25 == 0:
            logger.info("  evaluated %d / %d queries", i, len(valid_queries))

    metrics = evaluate_batch(valid_queries, retrieval_results, k_values)
    metrics.setdefault("metadata", {})["skipped_missing_target"] = skipped

    logger.info(
        "Scene-pool evaluation complete — MRR=%.4f, Acc@%d=%.3f",
        metrics["summary"]["mrr"],
        k_values[0],
        metrics["summary"][f"accuracy_at_{k_values[0]}"],
    )
    return metrics
