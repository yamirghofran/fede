"""Central configuration for the FEDE finetuning pipeline.

All paths, model identifiers, LLM settings, hyperparameters, and dataset
targets live here so that every other module in the package can import a
single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — resolved relative to this file so it works regardless of cwd
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
TAGGED_SCRIPTS_DIR = DATA_DIR / "scripts" / "parsed" / "tagged"
METADATA_PATH = DATA_DIR / "scripts" / "metadata" / "clean_parsed_meta.json"
FINETUNING_DATA_DIR = DATA_DIR / "finetuning"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_ID = "google/gemma-embedding-100m"
VECTOR_SIZE = 768

# Task prefixes (EmbeddingGemma convention)
QUERY_PREFIX = "Represent this query for retrieving relevant movie scenes: "
DOCUMENT_PREFIX = ""

# ---------------------------------------------------------------------------
# LLM — used only for synthetic data generation, not for retrieval
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
LLM_MODEL = os.getenv("FEDE_LLM_MODEL", "google/gemini-2.0-flash-lite")

LLM_MAX_RETRIES = 3
LLM_RATE_LIMIT_DELAY = 4  # seconds between calls
LLM_TEMPERATURE = 0.8
LLM_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Dataset generation targets
# ---------------------------------------------------------------------------
MIN_SCENE_WORDS = 40
TOP_SCENES_FOR_SUMMARY = 3
QUERIES_PER_MOVIE_SYNOPSIS = 4
RANDOM_NEGATIVES_PER_QUERY = 3
HARD_NEGATIVES_PER_QUERY = 3
POSITIVE_MIN_SCORE = 0.2
POSITIVE_CLOSE_GAP = 0.05
CHECKPOINT_INTERVAL = 50  # movies between checkpoints

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 16
WARMUP_RATIO = 0.1
CACHED_MNRL_MINI_BATCH = 64
MAX_QUERY_LENGTH = 96
MAX_DOCUMENT_LENGTH = 384

# LoRA fallback
USE_LORA = False
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_K_VALUES = [5, 10, 20]
EVAL_DATASET_SIZE = 100
EVAL_SENTENCE_POOL = 100
