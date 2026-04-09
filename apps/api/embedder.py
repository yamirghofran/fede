"""Hugging Face query embedding wrapper."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List

import numpy as np

from .settings import BackendSettings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

BASE_MODEL_NAME = "google/embeddinggemma-300m"

logger = logging.getLogger(__name__)


class QueryEmbedder:
    """Loads a SentenceTransformer model and encodes user queries."""

    def __init__(self, settings: BackendSettings, vector_size: int):
        self.settings = settings
        self.vector_size = vector_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self.load()
        return self._model

    def load(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        if self._model is not None:
            return self._model

        kwargs = {}
        hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        if hf_token:
            kwargs["token"] = hf_token
        if self.settings.embedding_device is not None:
            kwargs["device"] = self.settings.embedding_device
        if self.settings.embedding_fp16:
            import torch

            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

        from peft import PeftModel

        logger.info("Loading base model '%s' ...", BASE_MODEL_NAME)
        model = SentenceTransformer(BASE_MODEL_NAME, **kwargs)
        logger.info("Applying LoRA adapter '%s' ...", self.settings.embedding_model_id)
        model[0].auto_model = PeftModel.from_pretrained(
            model[0].auto_model, self.settings.embedding_model_id
        )
        dim = model.get_sentence_embedding_dimension()
        if dim is not None and dim != self.vector_size:
            raise ValueError(
                f"Embedding model dimension {dim} does not match Qdrant vector size "
                f"{self.vector_size}"
            )

        self._model = model
        logger.info("Embedding model ready on %s", model.device)
        return model

    def encode_query(self, query: str) -> List[float]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Query must not be blank")

        encode_fn = getattr(self.model, "encode_query", None)
        encode_kwargs = {
            "normalize_embeddings": True,
            "batch_size": self.settings.embedding_batch_size,
            "show_progress_bar": False,
        }
        if callable(encode_fn):
            embedding = encode_fn([cleaned], **encode_kwargs)
        else:
            embedding = self.model.encode([cleaned], prompt_name="query", **encode_kwargs)

        vector = np.asarray(embedding)[0]
        if vector.shape[0] != self.vector_size:
            raise ValueError(
                f"Encoded query dimension {vector.shape[0]} does not match Qdrant vector "
                f"size {self.vector_size}"
            )
        return vector.tolist()

    def info(self) -> dict:
        if self._model is None:
            return {
                "loaded": False,
                "model_id": self.settings.embedding_model_id,
                "vector_size": self.vector_size,
            }

        dim = self._model.get_sentence_embedding_dimension()
        return {
            "loaded": True,
            "model_id": self.settings.embedding_model_id,
            "vector_size": dim,
            "device": str(self._model.device),
        }
