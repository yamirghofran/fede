"""Assign positive scenes to synthetic queries via within-movie cosine similarity.

For each query, all scenes from the *same* movie are embedded with the base
(or round-1) model.  The scene with the highest cosine similarity is selected
as the positive.  A secondary positive is included when the score gap between
rank-1 and rank-2 is smaller than ``POSITIVE_CLOSE_GAP``.  Pairs whose best
score falls below ``POSITIVE_MIN_SCORE`` are discarded entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from preprocessing.chunker import SceneChunk

from finetuning.config import (
    DOCUMENT_PREFIX,
    POSITIVE_CLOSE_GAP,
    POSITIVE_MIN_SCORE,
    QUERY_PREFIX,
)

logger = logging.getLogger(__name__)


@dataclass
class PositiveMatch:
    """A query matched to one or more positive scenes."""

    query: str
    positives: List[SceneChunk]
    scores: List[float]


class PositiveAssigner:
    """Assigns the best-matching scene(s) within a movie for a given query.

    Holds a reference to a ``SentenceTransformer`` model so that repeated
    calls across movies reuse the same loaded weights.
    """

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    def assign(
        self,
        query: str,
        scenes: Sequence[SceneChunk],
        min_score: float = POSITIVE_MIN_SCORE,
        close_gap: float = POSITIVE_CLOSE_GAP,
    ) -> Optional[PositiveMatch]:
        """Find the best scene(s) for *query* within a single movie.

        Args:
            query: The synthetic search query.
            scenes: All qualifying scenes from the same movie.
            min_score: Discard the pair if the best score is below this.
            close_gap: Include a second positive when the gap between the
                top-2 scores is smaller than this threshold.

        Returns:
            A ``PositiveMatch`` with 1 or 2 positives, or ``None`` if no
            scene meets the minimum score.
        """
        if not scenes:
            return None

        query_emb = self._model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
        )
        doc_texts = [DOCUMENT_PREFIX + s.text for s in scenes]
        doc_embs = self._model.encode(
            doc_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        sims = np.dot(doc_embs, query_emb)

        ranked_idx = np.argsort(-sims)
        best_idx = ranked_idx[0]
        best_score = float(sims[best_idx])

        if best_score < min_score:
            return None

        positives = [scenes[best_idx]]
        scores = [best_score]

        if len(ranked_idx) > 1:
            second_idx = ranked_idx[1]
            second_score = float(sims[second_idx])
            if best_score - second_score < close_gap:
                positives.append(scenes[second_idx])
                scores.append(second_score)

        return PositiveMatch(query=query, positives=positives, scores=scores)

    def assign_batch(
        self,
        queries: List[str],
        scenes: Sequence[SceneChunk],
        min_score: float = POSITIVE_MIN_SCORE,
        close_gap: float = POSITIVE_CLOSE_GAP,
    ) -> List[PositiveMatch]:
        """Assign positives for multiple queries against the same scene set.

        Encodes the scene set once and reuses it across all queries.
        """
        if not scenes or not queries:
            return []

        doc_texts = [DOCUMENT_PREFIX + s.text for s in scenes]
        doc_embs = self._model.encode(
            doc_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        results: List[PositiveMatch] = []
        for query in queries:
            query_emb = self._model.encode(
                QUERY_PREFIX + query,
                normalize_embeddings=True,
            )
            sims = np.dot(doc_embs, query_emb)
            ranked_idx = np.argsort(-sims)
            best_idx = ranked_idx[0]
            best_score = float(sims[best_idx])

            if best_score < min_score:
                continue

            positives = [scenes[best_idx]]
            scores = [best_score]

            if len(ranked_idx) > 1:
                second_idx = ranked_idx[1]
                second_score = float(sims[second_idx])
                if best_score - second_score < close_gap:
                    positives.append(scenes[second_idx])
                    scores.append(second_score)

            results.append(PositiveMatch(query=query, positives=positives, scores=scores))

        return results
