"""Semantic retriever adapter for the evaluation pipeline.

Wraps ``ScriptRetriever.hierarchical_search()`` into the ``Retriever``
protocol expected by ``run_pipeline()``.  Encodes each query with the
fine-tuned model, runs hierarchical search, and deduplicates results
at the movie level.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from vector_db.config import QdrantConfig
from vector_db.retrieval import ScriptRetriever

from finetuning.config import EVAL_SENTENCE_POOL, QUERY_PREFIX

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Movie-level semantic retriever backed by Qdrant.

    Satisfies the ``Retriever`` protocol::

        def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]: ...
    """

    def __init__(
        self,
        model: SentenceTransformer,
        config: Optional[QdrantConfig] = None,
        sentence_pool: int = EVAL_SENTENCE_POOL,
    ) -> None:
        self._model = model
        self._retriever = ScriptRetriever(config)
        self._sentence_pool = sentence_pool

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Encode *query*, run hierarchical search, and deduplicate by movie.

        Returns a list of dicts sorted by descending score::

            [{"movie_id": str, "movie_title": str, "score": float}, ...]
        """
        query_emb = self._model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
        ).tolist()

        scene_results = self._retriever.hierarchical_search(
            query_embedding=query_emb,
            top_k=top_k * 3,  # over-fetch so dedup still yields >= top_k movies
            sentence_pool=self._sentence_pool,
        )

        # Deduplicate to movie level — keep the highest-scoring scene per movie
        seen: Dict[str, Dict[str, Any]] = {}
        for sr in scene_results:
            if sr.movie_id not in seen or sr.score > seen[sr.movie_id]["score"]:
                seen[sr.movie_id] = {
                    "movie_id": sr.movie_id,
                    "movie_title": sr.movie_title,
                    "score": sr.score,
                }

        ranked = sorted(seen.values(), key=lambda d: d["score"], reverse=True)
        return ranked[:top_k]
