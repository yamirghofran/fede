"""EmbeddingGemma model loading and task-prefix encoding helpers.

Provides a thin wrapper around ``SentenceTransformer`` that consistently
applies the query / document task prefixes defined in ``finetuning.config``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from finetuning.config import (
    DOCUMENT_PREFIX,
    EMBEDDING_MODEL_ID,
    MAX_DOCUMENT_LENGTH,
    MAX_QUERY_LENGTH,
    QUERY_PREFIX,
)

logger = logging.getLogger(__name__)


def load_model(
    model_id_or_path: Optional[str] = None,
    device: Optional[str] = None,
) -> SentenceTransformer:
    """Load a ``SentenceTransformer`` model.

    Args:
        model_id_or_path: HuggingFace model ID or local checkpoint path.
            Defaults to ``EMBEDDING_MODEL_ID`` from config.
        device: Force a specific device (``"cpu"``, ``"cuda"``, ``"mps"``).
            ``None`` lets Sentence Transformers auto-detect.

    Returns:
        A ready-to-use ``SentenceTransformer`` instance.
    """
    model_id = model_id_or_path or EMBEDDING_MODEL_ID
    logger.info("Loading embedding model: %s", model_id)

    kwargs = {}
    if device is not None:
        kwargs["device"] = device

    model = SentenceTransformer(model_id, **kwargs)
    logger.info("Model loaded — device=%s, dim=%d", model.device, model.get_sentence_embedding_dimension())
    return model


def encode_queries(
    model: SentenceTransformer,
    texts: Union[str, List[str]],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode query text(s) with the query task prefix.

    Prepends ``QUERY_PREFIX`` to every input and truncates to
    ``MAX_QUERY_LENGTH`` tokens.
    """
    if isinstance(texts, str):
        texts = [texts]
    prefixed = [QUERY_PREFIX + t for t in texts]
    return model.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )


def encode_documents(
    model: SentenceTransformer,
    texts: Union[str, List[str]],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode document text(s) with the document task prefix.

    Prepends ``DOCUMENT_PREFIX`` (empty string by default for
    EmbeddingGemma) and truncates to ``MAX_DOCUMENT_LENGTH`` tokens.
    """
    if isinstance(texts, str):
        texts = [texts]
    prefixed = [DOCUMENT_PREFIX + t for t in texts]
    return model.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
