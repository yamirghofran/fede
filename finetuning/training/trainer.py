"""Fine-tuning pipeline for the FEDE embedding model.

Sets up a ``SentenceTransformerTrainer`` with
``CachedMultipleNegativesRankingLoss`` and optional LoRA (via ``peft``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from finetuning.config import (
    CACHED_MNRL_MINI_BATCH,
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
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA fine-tuning.  Install with:  pip install peft"
        ) from exc

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
# Trainer factory
# ---------------------------------------------------------------------------

def build_trainer(
    model: SentenceTransformer,
    train_dataset: Dataset,
    output_dir: str,
    use_lora: bool = USE_LORA,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    warmup_ratio: float = WARMUP_RATIO,
    fp16: bool = True,
) -> SentenceTransformerTrainer:
    """Construct a ready-to-run ``SentenceTransformerTrainer``.

    Args:
        model: The embedding model (base or round-1 checkpoint).
        train_dataset: HuggingFace ``Dataset`` with ``anchor``,
            ``positive``, and optional ``negative_*`` columns.
        output_dir: Where to save checkpoints and the final model.
        use_lora: If ``True``, apply LoRA adapters before training.
        num_epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        learning_rate: Peak learning rate.
        warmup_ratio: Fraction of total steps used for LR warmup.
        fp16: Enable mixed-precision training.

    Returns:
        A ``SentenceTransformerTrainer`` — call ``.train()`` to start.
    """
    if use_lora:
        model = _apply_lora(model)

    loss = CachedMultipleNegativesRankingLoss(
        model=model,
        mini_batch_size=CACHED_MNRL_MINI_BATCH,
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        batch_sampler="no_duplicates",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=3,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    logger.info(
        "Trainer built — epochs=%d, batch=%d, lr=%.1e, lora=%s, output=%s",
        num_epochs, batch_size, learning_rate, use_lora, output_dir,
    )
    return trainer
