"""Central configuration for the FEDE finetuning pipeline.

All paths, model identifiers, LLM settings, hyperparameters, and dataset
targets live here so that every other module in the package can import a
single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root — resolved relative to this file so it works regardless of cwd
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

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
EMBEDDING_MODEL_ID = "google/embeddinggemma-300m"
VECTOR_SIZE = 768

# EmbeddingGemma-300m handles query/document prompts internally via
# encode_query() / encode_document() — no manual prefix prepending required.
QUERY_PREFIX = ""
DOCUMENT_PREFIX = ""

# Device for the embedding model during dataset building.
# Default is "cpu" to avoid OOM on Apple Silicon MPS (unified RAM).
# Set to "mps" or "cuda" only if you have enough dedicated VRAM.
FINETUNING_EMBED_DEVICE = os.getenv("FINETUNING_EMBED_DEVICE", "cpu")

# Batch size for encode_queries / encode_documents during dataset building.
# Keep small (8–16) to limit peak activation memory on CPU. Has no effect on
# embedding quality.
FINETUNING_ENCODE_BATCH_SIZE = int(os.getenv("FINETUNING_ENCODE_BATCH_SIZE", "8"))

# Load the embedding model in float16 to halve weight memory (~600 MB vs ~1.2 GB).
# Disable only if you see NaN embeddings (very unlikely for inference).
FINETUNING_EMBED_FP16 = os.getenv("FINETUNING_EMBED_FP16", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# LLM — used only for synthetic data generation, not for retrieval
#
# Two supported providers (set FEDE_LLM_PROVIDER in .env):
#
#   "openrouter"  — routes through openrouter.ai (default).
#                   Requires OPENROUTER_API_KEY.
#                   Cold-start latency common; ~60 req/min free tier.
#
#   "gemini"      — Google AI Studio direct API (OpenAI-compatible endpoint).
#                   Requires GEMINI_API_KEY.
#                   No cold starts; ~1-3 s latency per call.
#                   Free tier: 15 RPM → set FEDE_LLM_RATE_DELAY=4
#                   Paid tier: 2000 RPM → set FEDE_LLM_RATE_DELAY=0.05
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

GEMINI_BASE_URL      = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY_ENV   = "GEMINI_API_KEY"

_PROVIDER = os.getenv("FEDE_LLM_PROVIDER", "openrouter").lower()

if _PROVIDER == "gemini":
    LLM_BASE_URL     = GEMINI_BASE_URL
    LLM_API_KEY_ENV  = GEMINI_API_KEY_ENV
    LLM_MODEL        = os.getenv("FEDE_LLM_MODEL", "gemini-2.0-flash-lite")
    # Free tier default: 15 RPM = 1 req / 4 s.
    # If billing is enabled on your Google account, lower this to 0.05.
    _default_rate    = "4"
else:
    LLM_BASE_URL     = OPENROUTER_BASE_URL
    LLM_API_KEY_ENV  = OPENROUTER_API_KEY_ENV
    LLM_MODEL        = os.getenv("FEDE_LLM_MODEL", "google/gemini-2.5-flash-lite")
    _default_rate    = "1"

LLM_MAX_RETRIES = 3
LLM_RATE_LIMIT_DELAY = float(os.getenv("FEDE_LLM_RATE_DELAY", _default_rate))
LLM_TEMPERATURE = 0.8
LLM_MAX_TOKENS  = 1024
LLM_CONCURRENCY = int(os.getenv("FEDE_LLM_CONCURRENCY", "10"))

# ---------------------------------------------------------------------------
# Gemini Batch API — for bulk dataset generation (50% cheaper, no rate issues)
# ---------------------------------------------------------------------------
GEMINI_BATCH_MODEL = os.getenv("FEDE_BATCH_MODEL", "gemini-2.5-flash")
GEMINI_BATCH_POLL_INTERVAL = int(os.getenv("FEDE_BATCH_POLL_INTERVAL", "30"))

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
# 12 GB-class GPUs handle EmbeddingGemma training much more reliably with
# LoRA plus a modest per-device batch.
TRAIN_BATCH_SIZE = 4
WARMUP_RATIO = 0.1
# Cached MNRL uses an additional inner minibatch during loss computation.
# Keep this small on consumer GPUs to avoid activation spikes.
CACHED_MNRL_MINI_BATCH = 8
MAX_QUERY_LENGTH = 96
MAX_DOCUMENT_LENGTH = 384

# Prefer LoRA by default for consumer GPUs.
USE_LORA = True
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_K_VALUES = [5, 10, 20]
EVAL_DATASET_SIZE = 100
EVAL_SENTENCE_POOL = 100
