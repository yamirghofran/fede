"""EmbeddingGemma model loading and task-prefix encoding helpers.

Provides a thin wrapper around ``SentenceTransformer`` that consistently
applies the query / document task prefixes defined in ``finetuning.config``.
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from finetuning.config import (
    DOCUMENT_PREFIX,
    EMBEDDING_MODEL_ID,
    FINETUNING_EMBED_FP16,
    FINETUNING_ENCODE_BATCH_SIZE,
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
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if hf_token:
        kwargs["token"] = hf_token
    if device is not None:
        kwargs["device"] = device
    if FINETUNING_EMBED_FP16:
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

    model = SentenceTransformer(model_id, **kwargs)

    # Enforce a practical training-time truncation ceiling. Without this,
    # very long scene texts can silently drive up memory use and step time.
    max_seq_length = max(MAX_QUERY_LENGTH, MAX_DOCUMENT_LENGTH)
    model.max_seq_length = max_seq_length
    if hasattr(model[0], "max_seq_length"):
        model[0].max_seq_length = max_seq_length

    logger.info("Model loaded — device=%s, dim=%d", model.device, model.get_sentence_embedding_dimension())
    return model


def encode_queries(
    model: SentenceTransformer,
    texts: Union[str, List[str]],
    batch_size: int = FINETUNING_ENCODE_BATCH_SIZE,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode query text(s) using the model's query prompt.

    Returns shape ``(768,)`` for a single string, ``(N, 768)`` for a list.
    Uses ``encode_query`` when available (EmbeddingGemma-style models that
    declare query/document prompts internally), falls back to plain ``encode``.
    """
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    encode_fn = getattr(model, "encode_query", None)
    if callable(encode_fn):
        result = np.array(encode_fn(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        ))
    else:
        result = np.array(model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        ))
    return result[0] if single else result


def encode_documents(
    model: SentenceTransformer,
    texts: Union[str, List[str]],
    batch_size: int = FINETUNING_ENCODE_BATCH_SIZE,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode document text(s) using the model's document prompt.

    Returns shape ``(768,)`` for a single string, ``(N, 768)`` for a list.
    Uses ``encode_document`` when available (EmbeddingGemma-style models),
    falls back to plain ``encode``.
    """
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    encode_fn = getattr(model, "encode_document", None)
    if callable(encode_fn):
        result = np.array(encode_fn(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        ))
    else:
        result = np.array(model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        ))
    return result[0] if single else result
