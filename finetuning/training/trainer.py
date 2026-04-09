"""Fine-tuning pipeline for the FEDE embedding model.

Sets up a ``SentenceTransformerTrainer`` with
``CachedMultipleNegativesRankingLoss``, an optional
``InformationRetrievalEvaluator`` for per-epoch metrics, and optional
LoRA (via ``peft``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from finetuning.config import (
    CACHED_MNRL_MINI_BATCH,
    EVAL_K_VALUES,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGET_MODULES,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    USE_LORA,
    WARMUP_RATIO,
)

logger = logging.getLogger(__name__)


class FedeSentenceTransformerTrainer(SentenceTransformerTrainer):
    """Trainer with a LoRA-aware best-checkpoint restore path."""

    def _load_best_model(self) -> None:
        best_ckpt = getattr(self.state, "best_model_checkpoint", None)
        if not best_ckpt:
            return
        try:
            from finetuning.training.model import load_model_state

            logger.info("Loading best model state from %s", best_ckpt)
            load_model_state(self.model, best_ckpt)
        except Exception:
            logger.exception("Could not load the best model from %s", best_ckpt)


def _mixed_precision_flags(want_amp: bool, use_lora: bool) -> Tuple[bool, bool]:
    """Return ``(fp16, bf16)`` for ``TrainingArguments``.

    On CUDA, prefer **bf16** when supported: ``fp16`` + GradScaler with LoRA can
    raise ``ValueError: Attempting to unscale FP16 gradients``.
    If LoRA is on but bf16 is unavailable, fall back to **fp32** (avoid broken fp16 path).
    """
    if not want_amp:
        return False, False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return False, True
    if use_lora:
        logger.warning(
            "CUDA bfloat16 not available with LoRA — training in fp32 (reduce batch size if OOM)"
        )
        return False, False
    return True, False


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_training_dataset(jsonl_path: Path) -> Dataset:
    """Load a JSONL training file into a HuggingFace ``Dataset``.

    Each line must contain at least ``anchor`` and ``positive`` fields.
    An optional ``negatives`` list is included when present.
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            row = {"anchor": obj["anchor"], "positive": obj["positive"]}
            if obj.get("negatives"):
                for i, neg in enumerate(obj["negatives"]):
                    row[f"negative_{i}"] = neg
            rows.append(row)

    logger.info("Loaded %d training pairs from %s", len(rows), jsonl_path)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# LoRA wrapping
# ---------------------------------------------------------------------------

def _apply_lora(model: SentenceTransformer) -> SentenceTransformer:
    """Wrap the model's transformer backbone with LoRA adapters."""
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA fine-tuning.  Install with:  pip install peft"
        ) from exc

    if isinstance(model[0].auto_model, PeftModel):
        logger.info("Model already has LoRA adapters loaded; reusing existing PEFT model")
        return model

    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    inner = model[0].auto_model
    lora_model = get_peft_model(inner, peft_config)
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    logger.info(
        "LoRA applied — trainable: %d / %d (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )
    model[0].auto_model = lora_model
    return model


# ---------------------------------------------------------------------------
# Evaluator factory
# ---------------------------------------------------------------------------

def build_evaluator(
    eval_queries_path: Path,
    corpus: Dict[str, Any],
) -> InformationRetrievalEvaluator:
    """Build an ``InformationRetrievalEvaluator`` from the held-out eval set
    and the scene corpus.

    The evaluator runs after each epoch and computes MRR, NDCG, Accuracy,
    and Recall at the configured k values.

    Each scene is a separate corpus document so the evaluator embeds at the
    same granularity the model is trained on.  A query's relevant-doc set
    contains **all** scene IDs belonging to its target movie, so a hit on
    any scene from the correct movie counts as a success.

    Args:
        eval_queries_path: Path to ``eval_queries.json`` — each entry has
            ``query``, ``movie_id``, ``movie_title``.
        corpus: ``Dict[movie_id, MovieEntry]`` from ``build_scene_corpus()``.

    Returns:
        A ready-to-use ``InformationRetrievalEvaluator``.
    """
    with open(eval_queries_path, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)

    # queries: {qid: query_text}
    queries: Dict[str, str] = {}
    # relevant_docs: {qid: {doc_id, ...}}
    relevant_docs: Dict[str, Set[str]] = {}

    # Build a scene-level corpus and map each movie to its scene doc IDs.
    corpus_dict: Dict[str, str] = {}
    movie_scene_ids: Dict[str, Set[str]] = {}
    for movie_id, entry in corpus.items():
        scene_ids: Set[str] = set()
        for idx, scene in enumerate(entry.scenes):
            doc_id = f"{movie_id}__scene_{idx}"
            corpus_dict[doc_id] = scene.text
            scene_ids.add(doc_id)
        movie_scene_ids[movie_id] = scene_ids

    for i, q in enumerate(raw_queries):
        qid = f"q_{i}"
        queries[qid] = q["query"]
        relevant_docs[qid] = movie_scene_ids.get(q["movie_id"], set())

    max_k = max(EVAL_K_VALUES)
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus_dict,
        relevant_docs=relevant_docs,
        mrr_at_k=[max_k],
        ndcg_at_k=[max_k],
        accuracy_at_k=list(EVAL_K_VALUES),
        precision_recall_at_k=list(EVAL_K_VALUES),
        map_at_k=[max_k],
        batch_size=64,
        query_prompt_name="query",
        corpus_prompt_name="document",
        name="fede-ir-eval",
    )

    logger.info(
        "Evaluator built — %d queries, %d scene docs (from %d movies), k=%s",
        len(queries), len(corpus_dict), len(corpus), EVAL_K_VALUES,
    )
    return evaluator


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def build_trainer(
    model: SentenceTransformer,
    train_dataset: Dataset,
    output_dir: str,
    evaluator: Optional[InformationRetrievalEvaluator] = None,
    use_lora: bool = USE_LORA,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    warmup_ratio: float = WARMUP_RATIO,
    fp16: bool = True,
    cached_mnrl_mini_batch: Optional[int] = None,
) -> SentenceTransformerTrainer:
    """Construct a ready-to-run ``SentenceTransformerTrainer``.

    Args:
        model: The embedding model (base or round-1 checkpoint).
        train_dataset: HuggingFace ``Dataset`` with ``anchor``,
            ``positive``, and optional ``negative_*`` columns.
        output_dir: Where to save checkpoints and the final model.
        evaluator: Optional ``InformationRetrievalEvaluator`` — if
            provided, runs after each epoch and the best checkpoint is
            selected by MRR.
        use_lora: If ``True``, apply LoRA adapters before training.
        num_epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        learning_rate: Peak learning rate.
        warmup_ratio: Fraction of total steps used for LR warmup.
        fp16: Enable mixed-precision training (bf16 on supported CUDA GPUs, else fp16).
        cached_mnrl_mini_batch: Override ``CACHED_MNRL_MINI_BATCH`` from config (lower on limited VRAM).

    Returns:
        A ``SentenceTransformerTrainer`` — call ``.train()`` to start.
    """
    if use_lora:
        model = _apply_lora(model)

    mnrl_mb = cached_mnrl_mini_batch if cached_mnrl_mini_batch is not None else CACHED_MNRL_MINI_BATCH
    loss = CachedMultipleNegativesRankingLoss(
        model=model,
        mini_batch_size=mnrl_mb,
    )

    eval_strategy = "epoch" if evaluator else "no"
    load_best = evaluator is not None

    use_fp16, use_bf16 = _mixed_precision_flags(fp16, use_lora)

    # Map dataset columns to the model's named prompt templates so that
    # anchors are encoded with the "query" prompt and positives/negatives
    # with the "document" prompt during training — matching inference.
    prompts = {"anchor": "query", "positive": "document"}
    for col in train_dataset.column_names:
        if col.startswith("negative"):
            prompts[col] = "document"

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=use_fp16,
        bf16=use_bf16,
        batch_sampler="no_duplicates",
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_fede-ir-eval_cosine_mrr@20" if load_best else None,
        greater_is_better=True if load_best else None,
        prompts=prompts,
    )

    trainer = FedeSentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    prec = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")
    logger.info(
        "Trainer built — epochs=%d, batch=%d, mnrl_mini=%d, amp=%s, lr=%.1e, warmup=%.3f, lora=%s, evaluator=%s, output=%s",
        num_epochs, batch_size, mnrl_mb, prec, learning_rate, warmup_ratio, use_lora, evaluator is not None, output_dir,
    )
    return trainer
