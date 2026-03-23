"""EmbeddingGemma model loading and task-prefix encoding helpers.

Provides a thin wrapper around ``SentenceTransformer`` that consistently
applies the query / document task prefixes defined in ``finetuning.config``.
Also handles local LoRA/PEFT checkpoints explicitly so round-1/round-2
models can be reloaded across training, hard-negative mining, evaluation,
and indexing.
"""

from __future__ import annotations

import importlib
import json
import os
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer

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


def _apply_sequence_length(model: SentenceTransformer) -> SentenceTransformer:
    """Apply the configured truncation ceiling to a loaded model."""
    max_seq_length = max(MAX_QUERY_LENGTH, MAX_DOCUMENT_LENGTH)
    model.max_seq_length = max_seq_length
    if hasattr(model[0], "max_seq_length"):
        model[0].max_seq_length = max_seq_length
    if hasattr(model[0], "tokenizer"):
        model[0].tokenizer.model_max_length = max_seq_length
    return model


def _shared_hf_kwargs(device: Optional[str]) -> dict:
    """Build common Hugging Face load kwargs used for model restoration."""
    kwargs: dict = {}
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
    return kwargs


def _module_class(type_name: str):
    """Resolve a module class from the type entry in ``modules.json``."""
    module_name, class_name = type_name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _load_lora_sentence_transformer(model_dir: Path, device: Optional[str]) -> SentenceTransformer:
    """Manually reconstruct a SentenceTransformer from a LoRA checkpoint."""
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "peft is required to load LoRA fine-tuning checkpoints. Install with: pip install peft"
        ) from exc

    with open(model_dir / "adapter_config.json", "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    with open(model_dir / "modules.json", "r", encoding="utf-8") as f:
        module_defs = json.load(f)

    sb_cfg_path = model_dir / "sentence_bert_config.json"
    sb_cfg = {}
    if sb_cfg_path.exists():
        with open(sb_cfg_path, "r", encoding="utf-8") as f:
            sb_cfg = json.load(f)

    base_model_id = adapter_cfg["base_model_name_or_path"]
    shared_kwargs = _shared_hf_kwargs(device)
    model_args = dict(shared_kwargs.get("model_kwargs", {}))
    model_args["local_files_only"] = True
    tokenizer_args = {"local_files_only": True}
    config_args = {"local_files_only": True}
    if "token" in shared_kwargs:
        tokenizer_args["token"] = shared_kwargs["token"]
        config_args["token"] = shared_kwargs["token"]
    transformer = Transformer(
        model_name_or_path=base_model_id,
        max_seq_length=sb_cfg.get("max_seq_length"),
        do_lower_case=sb_cfg.get("do_lower_case", False),
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        config_args=config_args,
        tokenizer_name_or_path=str(model_dir),
    )
    transformer.auto_model = PeftModel.from_pretrained(
        transformer.auto_model,
        str(model_dir),
        is_trainable=True,
    )

    modules = [transformer]
    load_kwargs = {}
    if "token" in shared_kwargs:
        load_kwargs["token"] = shared_kwargs["token"]
    for module_def in module_defs[1:]:
        module_cls = _module_class(module_def["type"])
        subpath = module_def.get("path", "")
        if subpath:
            modules.append(module_cls.load(str(model_dir / subpath), **load_kwargs))
        else:
            modules.append(module_cls.load(**load_kwargs))

    model = SentenceTransformer(modules=modules, device=device)

    cfg_path = model_dir / "config_sentence_transformers.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model.prompts = cfg.get("prompts", model.prompts)
        model.default_prompt_name = cfg.get("default_prompt_name")
        model.similarity_fn_name = cfg.get("similarity_fn_name")

    return _apply_sequence_length(model)


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

    model_path = Path(model_id)
    if model_path.exists() and (model_path / "adapter_config.json").exists():
        model = _load_lora_sentence_transformer(model_path, device=device)
    else:
        kwargs = _shared_hf_kwargs(device)
        model = SentenceTransformer(model_id, **kwargs)
        model = _apply_sequence_length(model)

    logger.info("Model loaded — device=%s, dim=%d", model.device, model.get_sentence_embedding_dimension())
    return model


def load_model_state(target_model: SentenceTransformer, checkpoint_path: str) -> None:
    """Load a checkpoint into an existing model object in-place."""
    source_model = load_model(checkpoint_path, device=str(target_model.device))
    target_model.load_state_dict(source_model.state_dict(), strict=True)
    _apply_sequence_length(target_model)


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
