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
from typing import Any, Dict, List, Optional, Set

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from finetuning.config import (
    CACHED_MNRL_MINI_BATCH,
    DOCUMENT_PREFIX,
    EVAL_K_VALUES,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGET_MODULES,
    NUM_EPOCHS,
    QUERY_PREFIX,
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

    Args:
        eval_queries_path: Path to ``eval_queries.json`` — each entry has
            ``query``, ``movie_id``, ``movie_title``.
        corpus: ``Dict[movie_id, MovieEntry]`` from ``build_scene_corpus()``.
            Each movie's scenes are concatenated into one document per movie
            so that the evaluator measures movie-level retrieval.

    Returns:
        A ready-to-use ``InformationRetrievalEvaluator``.
    """
    with open(eval_queries_path, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)

    # queries: {qid: query_text}
    queries: Dict[str, str] = {}
    # relevant_docs: {qid: {doc_id, ...}}
    relevant_docs: Dict[str, Set[str]] = {}

    for i, q in enumerate(raw_queries):
        qid = f"q_{i}"
        queries[qid] = q["query"]
        relevant_docs[qid] = {q["movie_id"]}

    # corpus_dict: {doc_id: doc_text}
    # One document per movie — concatenate all scene texts so the evaluator
    # embeds each movie once rather than per-scene.
    corpus_dict: Dict[str, str] = {}
    for movie_id, entry in corpus.items():
        corpus_dict[movie_id] = "\n\n".join(s.text for s in entry.scenes)

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
        query_prompt=QUERY_PREFIX,
        corpus_prompt=DOCUMENT_PREFIX,
        name="fede-ir-eval",
    )

    logger.info(
        "Evaluator built — %d queries, %d corpus docs, k=%s",
        len(queries), len(corpus_dict), EVAL_K_VALUES,
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

    eval_strategy = "epoch" if evaluator else "no"
    load_best = evaluator is not None

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        batch_sampler="no_duplicates",
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_fede-ir-eval_cosine_mrr@20" if load_best else None,
        greater_is_better=True if load_best else None,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    logger.info(
        "Trainer built — epochs=%d, batch=%d, lr=%.1e, lora=%s, evaluator=%s, output=%s",
        num_epochs, batch_size, learning_rate, use_lora, evaluator is not None, output_dir,
    )
    return trainer
