# FEDE — Embedding Model Fine-tuning

This document covers the `finetuning/` package: a self-contained pipeline for generating training data, fine-tuning the embedding model on movie-script retrieval, and evaluating the result.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Module Structure](#3-module-structure)
4. [Prerequisites](#4-prerequisites)
5. [Configuration](#5-configuration)
6. [Pipeline Walkthrough](#6-pipeline-walkthrough)
   - [Step 1: Build the Training Dataset](#step-1-build-the-training-dataset)
   - [Step 2: Train Round 1](#step-2-train-round-1)
   - [Step 3: Mine Hard Negatives](#step-3-mine-hard-negatives)
   - [Step 4: Train Round 2](#step-4-train-round-2)
   - [Step 5: Index the Corpus](#step-5-index-the-corpus)
   - [Step 6: Evaluate](#step-6-evaluate)
7. [Training Dataset](#7-training-dataset)
   - [Query Types](#query-types)
   - [Positive Assignment](#positive-assignment)
   - [Negative Mining](#negative-mining)
   - [Output Schema](#output-schema)
8. [Training Details](#8-training-details)
   - [Loss Function](#loss-function)
   - [Per-Epoch Evaluation](#per-epoch-evaluation)
   - [LoRA Fallback](#lora-fallback)
9. [Evaluation](#9-evaluation)
   - [Metrics](#metrics)
   - [Eval Dataset Generation](#eval-dataset-generation)
   - [Retriever Modes](#retriever-modes)
10. [Environment Variables](#10-environment-variables)
11. [CLI Reference](#11-cli-reference)

---

## 1. Overview

The base embedding model (`google/embeddinggemma-300m`) was pre-trained on generic web text.  It produces reasonable similarity scores out of the box, but has never seen the specific mismatch at the core of FEDE: natural-language movie descriptions on the query side vs screenplay-formatted scene text on the document side.

Fine-tuning teaches the model this mapping through contrastive learning on synthetically generated query-scene pairs.  Training proceeds in two rounds:

1. **Round 1** — Train with synthetic queries and random negatives.  This gives the model a basic domain signal.
2. **Round 2** — Use the round-1 model to mine hard negatives (globally similar scenes from the wrong movie), then retrain.  This sharpens the decision boundary on the cases the model currently gets wrong.

---

## 2. Architecture

```
Tagged Scripts ──► ScriptChunker ──► Scene Corpus
                                         │
                  TMDB Metadata ─────────┤
                                         │
                                         ▼
                                   QueryGenerator ──► Type A (synopsis queries)
                                         │            Type B (scene-summary queries)
                                         │            Type C (paraphrases)
                                         ▼
                                  PositiveAssigner ──► within-movie cosine sim (Type A)
                                         │            direct source scene (Type B)
                                         ▼
                                   NegativeMiner ──► random negatives (round 1)
                                         │
                                         ▼
                               training_pairs_r1.jsonl
                                         │
                                         ▼
                             SentenceTransformerTrainer
                              CachedMNRL + EmbeddingGemma
                                         │
                                         ▼
                                   Round-1 Model
                                         │
                                         ▼
                              Hard Negative Mining ──► global corpus search
                                         │
                                         ▼
                               training_pairs_r2.jsonl
                                         │
                                         ▼
                             SentenceTransformerTrainer
                                         │
                                         ▼
                                 Final Fine-tuned Model
                                         │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                   index_corpus.py                 run_evaluation.py
                  (embed + upsert                (Accuracy@k, MRR)
                    to Qdrant)
```

---

## 3. Module Structure

```
finetuning/
├── __init__.py
├── config.py                        # All constants: paths, model ID, LLM config, hyperparams
│
├── corpus/
│   ├── __init__.py
│   └── scene_corpus.py              # ScriptChunker wrapper → per-movie scene list
│
├── dataset/
│   ├── __init__.py
│   ├── query_generator.py           # OpenRouter LLM: 3 query types, retry, leakage check
│   ├── positive_assigner.py         # Within-movie cosine similarity positive selection
│   ├── negative_miner.py            # Random negatives (r1) + hard negative mining (r2)
│   └── dataset_builder.py           # Orchestrator → JSONL export with checkpoint/resume
│
├── training/
│   ├── __init__.py
│   ├── model.py                     # EmbeddingGemma loader + task-prefix encode helpers
│   └── trainer.py                   # SentenceTransformerTrainer + CachedMNRL + evaluator + LoRA
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # accuracy_at_k, mean_reciprocal_rank, evaluate_batch
│   ├── dataset_generator.py         # Generates ~100 held-out eval queries via OpenRouter
│   ├── pipeline.py                  # run_pipeline(retriever, eval_path) → metrics dict
│   ├── semantic_retriever.py        # Wraps ScriptRetriever (Qdrant) into Retriever interface
│   └── memory_retriever.py          # In-memory brute-force retriever (no Qdrant needed)
│
└── scripts/
    ├── build_dataset.py             # CLI: build training dataset
    ├── mine_hard_negatives.py       # CLI: mine hard negatives with round-1 model
    ├── train.py                     # CLI: train round 1 or round 2
    ├── index_corpus.py              # CLI: embed all scenes + upsert to Qdrant
    └── run_evaluation.py            # CLI: run eval pipeline, print + save report
```

---

## 4. Prerequisites

- Python >= 3.9 with the project installed: `pip install -e .`
- DVC data pulled: `dvc pull` (provides tagged scripts and metadata)
- An OpenRouter API key (for synthetic query generation)
- Docker with Qdrant running (only needed for Qdrant-based evaluation and indexing)
- A GPU is recommended for training and corpus encoding, but not strictly required

---

## 5. Configuration

All settings live in `finetuning/config.py`.  The most important ones:

### Embedding Model

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_ID` | `google/embeddinggemma-300m` | HuggingFace model ID or local path |
| `HF_TOKEN` |  | (Optional) HuggingFace auth token if the embedding model is gated/private |
| `FINETUNING_EMBED_DEVICE` | `cpu` | Device for embedding model during dataset build. Use `cpu` on Apple Silicon to avoid MPS OOM |
| `FINETUNING_EMBED_FP16` | `true` | Load embedding model in float16 (~600 MB vs ~1.2 GB) |
| `FINETUNING_ENCODE_BATCH_SIZE` | `8` | Encode batch size during dataset build. Smaller = less peak activation memory |
| `TOKENIZERS_PARALLELISM` | `false` | Prevents joblib from forking worker processes (each fork copies the model into RAM) |
| `VECTOR_SIZE` | `768` | Output embedding dimension |
| `QUERY_PREFIX` | `""` | Not used — `embeddinggemma-300m` applies prompts internally via `encode_query()` / `encode_document()` |
| `DOCUMENT_PREFIX` | `""` | Not used (see above) |

### LLM (Synthetic Data Generation)

| Setting | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY_ENV` | `OPENROUTER_API_KEY` | Env var name for the API key |
| `LLM_MODEL` | `google/gemini-2.5-flash-lite` | Override with `FEDE_LLM_MODEL` env var |
| `LLM_RATE_LIMIT_DELAY` | `4` | Seconds between LLM calls |
| `LLM_TEMPERATURE` | `0.8` | Generation temperature |

### Dataset Targets

| Setting | Default | Description |
|---|---|---|
| `MIN_SCENE_WORDS` | `40` | Drop scenes shorter than this |
| `TOP_SCENES_FOR_SUMMARY` | `3` | Longest scenes per movie to summarise (Type B) |
| `QUERIES_PER_MOVIE_SYNOPSIS` | `4` | Synopsis queries per movie (Type A) |
| `RANDOM_NEGATIVES_PER_QUERY` | `3` | Random negatives per query (round 1) |
| `HARD_NEGATIVES_PER_QUERY` | `3` | Hard negatives per query (round 2) |
| `POSITIVE_MIN_SCORE` | `0.2` | Discard query-scene pairs below this cosine score |
| `POSITIVE_CLOSE_GAP` | `0.05` | Include a second positive if rank-1/rank-2 gap is smaller |

### Training Hyperparameters

| Setting | Default | Description |
|---|---|---|
| `LEARNING_RATE` | `2e-5` | Peak learning rate |
| `NUM_EPOCHS` | `2` | Training epochs |
| `TRAIN_BATCH_SIZE` | `16` | Per-device batch size |
| `WARMUP_RATIO` | `0.1` | Fraction of steps for LR warmup |
| `CACHED_MNRL_MINI_BATCH` | `64` | Sub-batch size for CachedMNRL |
| `USE_LORA` | `False` | Enable LoRA adapters (set `True` if GPU memory is tight) |

### Evaluation

| Setting | Default | Description |
|---|---|---|
| `EVAL_K_VALUES` | `[5, 10, 20]` | k values for Accuracy@k |
| `EVAL_DATASET_SIZE` | `100` | Number of held-out eval queries |

---

## 6. Pipeline Walkthrough

### Step 1: Build the Training Dataset

```bash
python -m finetuning.scripts.build_dataset --movies 1000
```

This parses tagged scripts, generates synthetic queries via the LLM, assigns positive scenes, adds random negatives, and writes the result to `data/finetuning/training_pairs_r1.jsonl`.

The process checkpoints every 50 movies.  If interrupted, re-run the same command — it resumes from the checkpoint automatically.

| Flag | Description |
|---|---|
| `--movies N` | Max movies to process (default: all) |
| `--output PATH` | Override output path |
| `--api-key KEY` | OpenRouter API key (or set `OPENROUTER_API_KEY`) |
| `--no-resume` | Start fresh, ignoring any checkpoint |

### Step 2: Train Round 1

```bash
python -m finetuning.scripts.train \
    --round 1 \
    --output fede-embeddinggemma/round1 \
    --eval-dataset data/finetuning/eval_queries.json
```

Trains the base model on the round-1 dataset with `CachedMultipleNegativesRankingLoss`.  If `--eval-dataset` is provided (or `data/finetuning/eval_queries.json` exists), an `InformationRetrievalEvaluator` runs after each epoch and the best checkpoint is automatically selected by MRR.

| Flag | Description |
|---|---|
| `--round 1\|2` | Training round (required) |
| `--model PATH` | Base model or checkpoint (default: `EMBEDDING_MODEL_ID`) |
| `--dataset PATH` | Training JSONL (default: `training_pairs_r{round}.jsonl`) |
| `--output DIR` | Checkpoint output directory (required) |
| `--eval-dataset PATH` | Eval queries for per-epoch evaluation |
| `--no-eval` | Skip per-epoch evaluation |
| `--lora` | Use LoRA adapters |
| `--epochs N` | Override epoch count |
| `--batch-size N` | Override batch size |
| `--lr FLOAT` | Override learning rate |

### Step 3: Mine Hard Negatives

```bash
python -m finetuning.scripts.mine_hard_negatives \
    --model fede-embeddinggemma/round1
```

Encodes the entire scene corpus with the round-1 model, then for each training query retrieves the globally top-scoring scenes from wrong movies.  These replace the random negatives to produce `training_pairs_r2.jsonl`.

| Flag | Description |
|---|---|
| `--model PATH` | Round-1 model checkpoint (required) |
| `--dataset PATH` | Round-1 JSONL (default: `training_pairs_r1.jsonl`) |
| `--output PATH` | Round-2 JSONL output (default: `training_pairs_r2.jsonl`) |
| `--movies N` | Max movies for corpus |
| `--hard-negs N` | Hard negatives per query (default: 3) |

### Step 4: Train Round 2

```bash
python -m finetuning.scripts.train \
    --round 2 \
    --model fede-embeddinggemma/round1 \
    --output fede-embeddinggemma/round2 \
    --eval-dataset data/finetuning/eval_queries.json
```

Same as round 1, but starts from the round-1 checkpoint and uses the hard-negative dataset.

### Step 5: Index the Corpus

```bash
python -m finetuning.scripts.index_corpus \
    --model fede-embeddinggemma/round2
```

Encodes every scene with the fine-tuned model and upserts into Qdrant.  Requires Docker Qdrant to be running.

| Flag | Description |
|---|---|
| `--model PATH` | Model to use for encoding (required) |
| `--movies N` | Max movies to index |
| `--batch-size N` | Embedding batch size (default: 64) |

### Step 6: Evaluate

```bash
# Against Qdrant (full end-to-end)
python -m finetuning.scripts.run_evaluation \
    --model fede-embeddinggemma/round2

# In-memory (no Qdrant needed — encodes corpus on the fly)
python -m finetuning.scripts.run_evaluation \
    --model fede-embeddinggemma/round2 --memory
```

| Flag | Description |
|---|---|
| `--model PATH` | Model to evaluate (required) |
| `--memory` | Use in-memory retriever instead of Qdrant |
| `--movies N` | Max movies for in-memory corpus |
| `--eval-dataset PATH` | Override eval queries path |
| `--report PATH` | Override report output path |

---

## 7. Training Dataset

### Query Types

The dataset is built from three sources of synthetic supervision:

**Type A — Synopsis queries (40% of data)**

Given the TMDB `overview` for a movie, the LLM generates 4 diverse search queries (plot-focused, conflict-focused, relationship-focused, theme-focused).  These are matched to their best scene within the same movie via embedding similarity.

**Type B — Scene-summary queries (40% of data)**

The 3 longest scenes per movie are individually converted by the LLM into 1-sentence plain-English search queries.  The source scene is used directly as the positive — no embedding-based assignment needed.  This is the highest-quality signal because it exactly models the mismatch the model needs to learn: natural language query vs screenplay-formatted document.

**Type C — Paraphrases (20% of data)**

A sample of existing Type A and B queries are rephrased by the LLM in 2 alternative wordings.  This cheaply multiplies data volume and teaches the model to be invariant to surface-level phrasing differences.

### Positive Assignment

- **Type A**: The base model encodes the query and all scenes from the same movie.  The scene with the highest cosine similarity is selected as the positive.  A second positive is included if the score gap between rank 1 and rank 2 is less than 0.05.  Pairs whose best score is below 0.2 are discarded.
- **Type B**: The scene that was summarised is the positive by construction — no assignment needed.
- **Type C**: Inherits the positive from the source query.

### Negative Mining

**Round 1 — Random negatives:** 3 scenes sampled uniformly from other movies.  Cheap and easy; provides a basic contrastive signal.

**Round 2 — Hard negatives:** The round-1 model encodes the full corpus.  For each query, the globally top-scoring scenes that belong to a different movie are selected as negatives.  These are the cases the round-1 model currently confuses, making the round-2 training signal much sharper.

### Output Schema

Each line of the JSONL file:

```json
{
  "anchor": "A retired detective is drawn back into a cold case that mirrors his own past.",
  "positive": "INT. DETECTIVE'S OFFICE - NIGHT\nThe desk is covered in old photographs...",
  "negatives": ["<scene from movie X>", "<scene from movie Y>", "<scene from movie Z>"],
  "movie_id": "some_movie",
  "movie_title": "Some Movie",
  "query_type": "synopsis"
}
```

The `movie_id`, `movie_title`, and `query_type` fields are metadata for debugging and traceability.  The Sentence Transformers trainer only reads `anchor`, `positive`, and `negative_*` columns.

### Leakage Detection

All generated queries are checked for title leakage before inclusion.  A query is rejected if more than half of the movie title's significant (non-stopword) words appear in the query text.

---

## 8. Training Details

### Loss Function

The model is trained with `CachedMultipleNegativesRankingLoss` (CachedMNRL).  For each query in a batch, the loss requires the model to rank the correct scene higher than all other scenes in the batch (in-batch negatives) plus any explicit hard negatives.

The "Cached" variant computes embeddings in sub-batches of 64 and caches them, enabling a much larger effective batch size than what fits in GPU memory at once.  Larger effective batches = more negatives per query = stronger training signal.

### Per-Epoch Evaluation

When an eval dataset is available, an `InformationRetrievalEvaluator` runs after each epoch.  It encodes all eval queries and all movies (scene text concatenated per movie), computes cosine similarity, and reports:

- MRR@20
- NDCG@20
- Accuracy@5, @10, @20
- Precision and Recall@5, @10, @20
- MAP@20

The trainer uses `load_best_model_at_end=True` with `metric_for_best_model="eval_fede-ir-eval_cosine_mrr@20"`, so the checkpoint with the highest MRR is automatically selected as the final model.

### LoRA Fallback

If GPU memory is too limited for full fine-tuning, pass `--lora` to the training script.  This wraps the transformer backbone with LoRA adapters (`r=16`, targeting `q_proj` and `v_proj`) so that only ~0.5% of parameters are updated.  Set `USE_LORA = True` in `config.py` to make this the default.

---

## 9. Evaluation

### Metrics

| Metric | Description |
|---|---|
| **Accuracy@k** | Fraction of queries where the correct movie appears in the top k results |
| **MRR** | Mean Reciprocal Rank — average of 1/rank across all queries.  MRR=1.0 means the correct movie is always rank 1 |
| **Median Rank** | The median position of the correct movie across all queries |

### Eval Dataset Generation

The held-out eval dataset is generated separately from the training data.  It consists of ~100 movie-level queries from movies that were *not* used in training.  Each query is a 1-sentence whole-movie description generated by the LLM.

Generate it before training:

```bash
# From Python:
from finetuning.corpus.scene_corpus import build_scene_corpus, load_metadata
from finetuning.evaluation.dataset_generator import generate_eval_dataset

corpus = build_scene_corpus(max_movies=1000)
metadata = load_metadata()
generate_eval_dataset(
    corpus_movie_ids=set(corpus.keys()),
    metadata=metadata,
)
```

### Retriever Modes

The evaluation pipeline accepts any object with a `.retrieve(query, top_k)` method.  Two implementations are provided:

**`SemanticRetriever`** — Queries Qdrant via `ScriptRetriever.hierarchical_search()`.  This is the full end-to-end path: sentence search, scene resolution, score merging, movie-level deduplication.  Requires Qdrant to be running and the corpus to be indexed.

**`MemoryRetriever`** — Encodes the entire scene corpus into a numpy matrix and answers queries via brute-force cosine similarity.  Produces identical movie-level rankings (assuming the same model and corpus) but needs no infrastructure.  Use this for fast iteration during development.

---

## 10. Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | API key for OpenRouter (required for dataset generation) |
| `FEDE_LLM_MODEL` | Override the LLM model used for query generation (default: `google/gemini-2.0-flash-lite`) |
| `QDRANT_*` | All Qdrant settings — see `docs/vector_store.md` for the full list |

Importing `finetuning.config` loads the project-root **`.env`** via `python-dotenv`, so you can keep `OPENROUTER_API_KEY` in `.env` without exporting it in the shell. You can still override with `export` or `--api-key`.

---

## 11. CLI Reference

All scripts are invoked as Python modules from the project root:

```bash
# Build round-1 training dataset
python -m finetuning.scripts.build_dataset --movies 1000

# Train round 1 (with per-epoch eval)
python -m finetuning.scripts.train \
    --round 1 \
    --output fede-embeddinggemma/round1 \
    --eval-dataset data/finetuning/eval_queries.json

# Mine hard negatives using round-1 model
python -m finetuning.scripts.mine_hard_negatives \
    --model fede-embeddinggemma/round1

# Train round 2 (with hard negatives)
python -m finetuning.scripts.train \
    --round 2 \
    --model fede-embeddinggemma/round1 \
    --output fede-embeddinggemma/round2 \
    --eval-dataset data/finetuning/eval_queries.json

# Index corpus into Qdrant with fine-tuned model
python -m finetuning.scripts.index_corpus \
    --model fede-embeddinggemma/round2

# Evaluate against Qdrant
python -m finetuning.scripts.run_evaluation \
    --model fede-embeddinggemma/round2

# Evaluate in-memory (no Qdrant)
python -m finetuning.scripts.run_evaluation \
    --model fede-embeddinggemma/round2 --memory
```

### Expected Data Volumes

For 1,000 movies:

| Data | Approximate count |
|---|---|
| Type A queries | 4,000 |
| Type B queries | 3,000 |
| Type C paraphrases | ~1,400 |
| **Round 1 total** | **~8,400 pairs** |
| Round 2 total | same pairs with hard negatives |
| Eval set | ~100 queries (held-out movies) |
