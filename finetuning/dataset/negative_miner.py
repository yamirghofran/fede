"""Negative scene mining for contrastive training.

Provides two strategies:

* **Random negatives** (round 1): sample scenes from movies other than the
  query's source movie.  Cheap and easy — gives the model a basic
  contrastive signal.
* **Hard negatives** (round 2): encode the full corpus with the round-1
  model, then for each query retrieve the globally top-scoring scenes that
  do *not* belong to the correct movie.  These are the cases the round-1
  model currently gets wrong, making the round-2 signal much sharper.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from preprocessing.chunker import SceneChunk

from finetuning.config import (
    HARD_NEGATIVES_PER_QUERY,
    RANDOM_NEGATIVES_PER_QUERY,
)
from finetuning.corpus.scene_corpus import MovieEntry
from finetuning.training.model import encode_documents, encode_queries

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Random negatives (round 1)
# ---------------------------------------------------------------------------

def sample_random_negatives(
    movie_id: str,
    corpus: Dict[str, MovieEntry],
    n: int = RANDOM_NEGATIVES_PER_QUERY,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Pick *n* random scene texts from movies other than *movie_id*.

    Uses reservoir-style sampling across the entire corpus so every
    scene in every other movie has equal probability.
    """
    rng = rng or random.Random()
    pool: List[str] = []
    for mid, entry in corpus.items():
        if mid == movie_id:
            continue
        pool.extend(s.text for s in entry.scenes)

    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


# ---------------------------------------------------------------------------
# Hard negatives (round 2)
# ---------------------------------------------------------------------------

@dataclass
class CorpusIndex:
    """Pre-computed embedding matrix for the entire scene corpus.

    Built once and reused across all queries during hard-negative mining.
    """

    embeddings: np.ndarray          # (N, dim) float32
    movie_ids: List[str]            # length N — movie_id per row
    scene_texts: List[str]          # length N — scene text per row

    @classmethod
    def build(
        cls,
        corpus: Dict[str, MovieEntry],
        model: SentenceTransformer,
        batch_size: int = 128,
    ) -> "CorpusIndex":
        """Encode every scene in the corpus and store the results."""
        texts: List[str] = []
        mids: List[str] = []

        for mid, entry in corpus.items():
            for scene in entry.scenes:
                texts.append(scene.text)
                mids.append(mid)

        logger.info("Encoding %d scenes for hard-negative index …", len(texts))
        embs = encode_documents(model, texts, batch_size=batch_size, show_progress=True)
        return cls(embeddings=embs, movie_ids=mids, scene_texts=texts)


def mine_hard_negatives(
    query: str,
    movie_id: str,
    index: CorpusIndex,
    model: SentenceTransformer,
    n: int = HARD_NEGATIVES_PER_QUERY,
) -> List[str]:
    """Retrieve the top-*n* globally closest scenes that belong to a different movie.

    The query is encoded with the task prefix; dot-product against the
    pre-computed corpus index yields cosine similarity (both sides are
    L2-normalised).
    """
    query_emb = encode_queries(model, query)
    sims = np.dot(index.embeddings, query_emb)
    ranked = np.argsort(-sims)

    negatives: List[str] = []
    for idx in ranked:
        if index.movie_ids[idx] == movie_id:
            continue
        negatives.append(index.scene_texts[idx])
        if len(negatives) >= n:
            break

    return negatives
