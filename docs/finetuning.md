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
   - [Step 1a: Generate Queries](#step-1a-generate-queries)
   - [Step 1b: Clean Raw Queries](#step-1b-clean-raw-queries)
   - [Step 1c: Assemble Training Pairs](#step-1c-assemble-training-pairs)
   - [Step 2: Train Round 1](#step-2-train-round-1)
   - [Step 3: Mine Hard Negatives](#step-3-mine-hard-negatives)
   - [Step 4: Train Round 2](#step-4-train-round-2)
   - [Step 5: Evaluate](#step-5-evaluate)
   - [Step 6: Index the Corpus](#step-6-index-the-corpus)
7. [Training Dataset](#7-training-dataset)
   - [Query Types](#query-types)
   - [Query Cleaning & Preprocessing](#query-cleaning--preprocessing)
   - [Positive Assignment](#positive-assignment)
   - [Negative Mining](#negative-mining)
   - [Output Schema](#output-schema)
8. [Training Details](#8-training-details)
   - [Loss Function](#loss-function)
   - [Prompt Alignment](#prompt-alignment)
   - [Per-Epoch Evaluation](#per-epoch-evaluation)
   - [LoRA Fallback](#lora-fallback)
9. [Evaluation](#9-evaluation)
   - [Metrics](#metrics)
   - [Scene-Pool Evaluation](#scene-pool-evaluation)
   - [Eval Dataset Generation](#eval-dataset-generation)
10. [v1 Results & Issues](#10-v1-results--issues)
    - [Original Results (v1)](#original-results-v1)
    - [Issues Identified](#issues-identified)
    - [Fixes Applied (v2)](#fixes-applied-v2)
11. [Environment Variables](#11-environment-variables)
12. [CLI Reference](#12-cli-reference)

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
│   ├── query_generator.py           # OpenRouter/Gemini LLM: 3 query types, retry, leakage check
│   ├── positive_assigner.py         # Within-movie cosine similarity positive selection
│   ├── negative_miner.py            # Random negatives (r1) + hard negative mining (r2)
│   └── dataset_builder.py           # Orchestrator → JSONL export with checkpoint/resume
│
├── training/
│   ├── __init__.py
│   ├── model.py                     # EmbeddingGemma loader + task-prefix encode helpers (bf16 safe)
│   └── trainer.py                   # SentenceTransformerTrainer + CachedMNRL + evaluator + LoRA
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # accuracy_at_k, mean_reciprocal_rank, evaluate_batch
│   ├── dataset_generator.py         # Generates held-out eval queries (scene-based or synopsis)
│   ├── scene_evaluator.py           # Scene-level max-pool evaluation (no Qdrant required)
│   ├── pipeline.py                  # run_pipeline(retriever, eval_path) → metrics dict
│   ├── semantic_retriever.py        # Wraps ScriptRetriever (Qdrant) into Retriever interface
│   └── memory_retriever.py          # In-memory brute-force retriever (no Qdrant needed)
│
└── scripts/
    ├── build_dataset.py             # CLI: build training dataset
    ├── clean_raw_queries.py         # CLI: repair, score, deduplicate raw LLM queries
    ├── generate_eval.py             # CLI: generate scene-based eval dataset
    ├── mine_hard_negatives.py       # CLI: mine hard negatives with round-1 model
    ├── train.py                     # CLI: train round 1 or round 2
    └── index_corpus.py              # CLI: embed all scenes + sentences, upsert to Qdrant
```

---

## 4. Prerequisites

- Python >= 3.12 with the project installed: `pip install -e .`
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
| `LLM_RATE_LIMIT_DELAY` | `1` | Min seconds between successive LLM calls (global). Override with `FEDE_LLM_RATE_DELAY`. Raise to `2`–`4` if you see 429s. |
| `LLM_CONCURRENCY` | `5` | Movies processed concurrently. Override with `FEDE_LLM_CONCURRENCY`. |
| `LLM_TEMPERATURE` | `0.8` | Generation temperature |

### Dataset Targets

| Setting | Default | Description |
|---|---|---|
| `MIN_SCENE_WORDS` | `40` | Drop scenes shorter than this |
| `TOP_SCENES_FOR_SUMMARY` | `3` | Longest scenes per movie to summarise (Type B) |
| `QUERIES_PER_MOVIE_SYNOPSIS` | `4` | Synopsis queries per movie (Type A) |
| `RANDOM_NEGATIVES_PER_QUERY` | `3` | Random negatives per query (round 1) |
| `HARD_NEGATIVES_PER_QUERY` | `3` | Hard negatives per query (round 2) |
| `POSITIVE_MIN_SCORE` | `0.35` | Discard query-scene pairs below this cosine score (raised from 0.2 in v2 — see [Issues](#issues-identified)) |
| `POSITIVE_CLOSE_GAP` | `0.05` | Include a second positive if rank-1/rank-2 gap is smaller |

### Training Hyperparameters

| Setting | Default | Description |
|---|---|---|
| `LEARNING_RATE` | `2e-5` | Peak learning rate (Round 1) |
| `ROUND2_LEARNING_RATE` | `5e-6` | Peak learning rate for Round 2 (4x lower to prevent catastrophic forgetting) |
| `NUM_EPOCHS` | `2` | Training epochs |
| `TRAIN_BATCH_SIZE` | `16` | Per-device batch size |
| `WARMUP_RATIO` | `0.1` | Fraction of total steps for linear LR warmup |
| `CACHED_MNRL_MINI_BATCH` | `8` | Sub-batch size for CachedMNRL (lower = less VRAM) |
| `MAX_DOCUMENT_LENGTH` | `512` | Max token length for document truncation (increased from 384 in v2) |
| `MAX_QUERY_LENGTH` | `96` | Max token length for query truncation |
| `USE_LORA` | `True` | Enable LoRA adapters (recommended for 12 GB GPUs) |

### Evaluation

| Setting | Default | Description |
|---|---|---|
| `EVAL_K_VALUES` | `[5, 10, 20]` | k values for Accuracy@k |
| `EVAL_DATASET_SIZE` | `100` | Number of held-out eval queries |

---

## 6. Pipeline Walkthrough

The notebook (`notebooks/finetuning_pipeline.ipynb`) executes each step as a separate cell.  The same operations are available as CLI scripts.

### Step 1a: Generate Queries

```bash
python -m finetuning.scripts.build_dataset --movies 1200
```

Parses tagged scripts, generates synthetic queries via the LLM (Type A synopsis, Type B scene-summary, Type C paraphrases), and writes `data/finetuning/raw_queries.jsonl`.

**Concurrency:** Movies are processed `LLM_CONCURRENCY` at a time using `asyncio`.  A shared rate-limiter enforces `LLM_RATE_LIMIT_DELAY` seconds between successive API calls.  If you see 429 errors, raise `FEDE_LLM_RATE_DELAY` to `2`–`4` in `.env`.

The process checkpoints every 50 movies.  If interrupted, re-run the same command — it resumes automatically.

### Step 1b: Clean Raw Queries

```bash
python -m finetuning.scripts.clean_raw_queries
```

LLM-generated queries are noisy.  Common issues include:
- **JSON wrapper artifacts** — the LLM sometimes returns `{"query": "..."}` instead of plain text
- **Markdown fences** — responses wrapped in ` ```json ... ``` `
- **Truncated output** — queries cut off mid-sentence due to token limits
- **Title/character leakage** — queries that name the movie or its characters, defeating the retrieval task
- **Generic phrasing** — vague queries like "a man discovers a secret" that match dozens of movies
- **Near-duplicates** — multiple queries with the same content tokens across movies

The cleaner addresses each of these through a multi-stage process:

1. **Structural repair** — strips markdown fences, parses JSON wrappers, salvages partial JSON, normalises whitespace and quotes.
2. **Quality scoring** — each query receives a 0.0–1.0 score based on penalties for leakage, truncation, banned prefixes, generic phrasing, abstract theme wording, and excessive length.  Scene-summary queries require ≥ 0.75 to pass; synopsis queries require ≥ 0.85.
3. **LLM regeneration** — queries that fail the quality threshold but have a recoverable source (the original scene text or movie synopsis) are regenerated with a fresh LLM call.
4. **Deduplication** — exact and near-duplicate removal using Jaccard similarity (threshold 0.85) on content tokens, first within each movie then globally.

**Output files:**

| File | Contents |
|---|---|
| `raw_queries.cleaned.jsonl` | Surviving queries — used by pair assembly |
| `raw_queries.audit.jsonl` | Full audit trail for every row (keep/regenerate/reject + reasons) |
| `raw_queries.rejected.jsonl` | Rejected rows with reasons |
| `raw_queries.cleaning_report.json` | Summary statistics |

### Step 1c: Assemble Training Pairs

Runs automatically as part of the notebook's Stage 1b cell, or via `DatasetBuilder.assemble_pairs()`.

Reads `raw_queries.cleaned.jsonl` (falls back to `raw_queries.jsonl` if the cleaned file does not exist), loads the embedding model, runs `PositiveAssigner` for Type A queries, samples random negatives, and writes `training_pairs_r1.jsonl`.

This step applies `POSITIVE_MIN_SCORE` (0.35) to filter low-confidence query-scene matches.  If you change this threshold, you must **delete** `training_pairs_r1.jsonl` before re-running — the resume logic detects already-processed movies and will skip them otherwise.

### Step 2: Train Round 1

```bash
python -m finetuning.scripts.train \
    --round 1 \
    --output fede-embeddinggemma/round1 \
    --eval-dataset data/finetuning/eval_queries.json
```

Trains the base model on the round-1 dataset with `CachedMultipleNegativesRankingLoss`.  If an eval dataset exists, a scene-level `InformationRetrievalEvaluator` runs after each epoch and the best checkpoint is selected by MRR@20.

Uses `LEARNING_RATE` (2e-5) with `WARMUP_RATIO` (10% linear warmup).

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

Encodes the entire scene corpus with the round-1 model using `encode_documents()` (which applies the model's native document prompt), then for each training query retrieves the globally top-scoring scenes from wrong movies via `encode_queries()`.  These replace the random negatives to produce `training_pairs_r2.jsonl`.

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
    --eval-dataset data/finetuning/eval_queries.json \
    --lr 5e-6
```

Same as Round 1, but starts from the Round 1 checkpoint, uses the hard-negative dataset, and defaults to `ROUND2_LEARNING_RATE` (5e-6) — 4x lower than Round 1 to prevent catastrophic forgetting of the domain signal learned in the first round.

### Step 5: Evaluate

Evaluation uses a **scene-level max-pool** strategy (see [Scene-Pool Evaluation](#scene-pool-evaluation)) that requires no Qdrant infrastructure.  This is run directly in the notebook's Stage 6 cell via `run_scene_eval()`.

```bash
# Generate the eval dataset (once)
python -m finetuning.scripts.generate_eval --scenes-per-movie 3

# Evaluation is done in the notebook — no separate CLI step needed
```

### Step 6: Index the Corpus

```bash
python -m finetuning.scripts.index_corpus \
    --model fede-embeddinggemma/round2
```

Encodes every scene **and sentence** with the fine-tuned model and upserts into Qdrant.  Requires Docker Qdrant to be running.  Sentences are filtered to only include those belonging to scenes that pass the `MIN_SCENE_WORDS` threshold.

| Flag | Description |
|---|---|
| `--model PATH` | Model to use for encoding (required) |
| `--movies N` | Max movies to index |
| `--batch-size N` | Embedding batch size (default: 64) |
| `--scenes-only` | Skip sentence encoding and upsert |

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

### Query Cleaning & Preprocessing

Raw LLM output is unreliable.  The `clean_raw_queries.py` script transforms `raw_queries.jsonl` into `raw_queries.cleaned.jsonl` through:

1. **Structural repair** — unwrap JSON artifacts (`{"query": "..."}`) and markdown fences, normalise unicode quotes and whitespace, reject truncated output (unmatched braces, trailing conjunctions, ellipses).
2. **Quality scoring** — a 0–1 score penalises title/character leakage (−0.20 each), generic subjects without distinctive events (−0.20), banned synopsis prefixes like "looking for" (−0.25), questions ("what happens when…", −0.10), abstract theme wording (−0.10), overstuffed clauses (−0.15), and length violations (−0.15).
3. **Regeneration** — queries below the keep threshold (0.75 for scene summaries, 0.85 for synopsis) are re-generated from their source scene text or movie synopsis via a fresh LLM call.
4. **Deduplication** — exact matches (by normalised content tokens) and near-duplicates (Jaccard ≥ 0.85) are removed, first within each movie then globally.

**Why this matters:** without cleaning, the training dataset contains ~15–20% noisy pairs — queries that leak the answer, match the wrong scene, or are too vague to discriminate between movies.  These degrade the contrastive loss by teaching the model to associate poor queries with good scenes.

### Positive Assignment

- **Type A**: The base model encodes the query and all scenes from the same movie.  The scene with the highest cosine similarity is selected as the positive.  A second positive is included if the score gap between rank 1 and rank 2 is less than `POSITIVE_CLOSE_GAP` (0.05).  Pairs whose best score is below `POSITIVE_MIN_SCORE` (0.35) are discarded as low-confidence matches.
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

The "Cached" variant computes embeddings in sub-batches (default 8) and caches them, enabling a much larger effective batch size than what fits in GPU memory at once.  Larger effective batches = more negatives per query = stronger training signal.

### Prompt Alignment

EmbeddingGemma-300m has 14 built-in prompt templates.  The two relevant for retrieval are:

| Prompt name | Template prefix |
|---|---|
| `query` | `task: search result \| query: ` |
| `document` | `title: none \| text: ` |

The trainer maps dataset columns to these prompts via `SentenceTransformerTrainingArguments.prompts`:

```python
prompts = {
    "anchor": "query",
    "positive": "document",
    "negative_0": "document",
    "negative_1": "document",
    "negative_2": "document",
}
```

This ensures that during training the model encodes queries and documents with the same prefixes it uses at inference.  In v1, this mapping was missing — training encoded all columns without any prompt, while evaluation used the native prompts.  See [Issues Identified](#issues-identified) for the impact.

### Per-Epoch Evaluation

When an eval dataset is available, an `InformationRetrievalEvaluator` runs after each epoch using a **scene-level corpus** — each scene is a separate document, and a query's relevant-doc set contains all scene IDs for its target movie.  This aligns the in-training evaluation with the scene-level training objective.

The evaluator reports MRR@20, NDCG@20, Accuracy@5/10/20, Precision/Recall@5/10/20, and MAP@20.  The trainer uses `load_best_model_at_end=True` with `metric_for_best_model="eval_fede-ir-eval_cosine_mrr@20"`, so the checkpoint with the highest MRR is automatically selected.

### LoRA Fallback

LoRA is the default mode for 12 GB GPUs.  It wraps the transformer backbone with adapters (`r=16`, targeting `q_proj` and `v_proj`) so that only ~0.32% of parameters are updated, keeping VRAM usage under ~4–5 GB during training.

Full fine-tuning can be enabled by setting `USE_LORA = False`, but requires significantly more VRAM.

---

## 9. Evaluation

### Metrics

| Metric | Description |
|---|---|
| **Accuracy@k** | Fraction of queries where the correct movie appears in the top k results |
| **MRR** | Mean Reciprocal Rank — average of 1/rank across all queries.  MRR=1.0 means the correct movie is always rank 1 |
| **Median Rank** | The median position of the correct movie across all queries |

### Scene-Pool Evaluation

The primary evaluation method is **scene-level max-pool** (`scene_evaluator.py`), which requires no Qdrant infrastructure:

1. All scenes in the corpus (~123k from ~1,262 movies) are encoded using `encode_documents()`.
2. Each eval query is encoded using `encode_queries()`.
3. For each query, cosine similarity is computed against every scene.  The movie-level score is `max(cosine_sim)` across all of a movie's scenes.
4. Movies are ranked by this max-scene score, and Accuracy@k / MRR are computed.

This aligns evaluation with the training objective — the model learns to match queries to individual scenes, so evaluation should rank movies by their best-matching scene rather than by a truncated concatenation of all scenes.

Queries whose target movie is not present in the corpus are automatically excluded from metrics.

### Eval Dataset Generation

The held-out eval dataset consists of ~184 scene-based queries from ~62 movies that were **not** used in training.  Each query is generated by the LLM from an actual scene in the eval movie, producing queries that directly test the scene-retrieval capability.

```bash
python -m finetuning.scripts.generate_eval --scenes-per-movie 3 --seed 42
```

The generator:
1. Builds the full scene corpus and the training subset (1,200 movies).
2. Identifies the eval pool (all movies not in the training set).
3. For each eval movie, picks up to 3 of the longest scenes and generates one query per scene.
4. Backs up any existing `eval_queries.json` before overwriting.

---

## 10. v1 Results & Issues

### Original Results (v1)

The first fine-tuning run (v1) used 1,200 training movies with the following settings: `LEARNING_RATE=2e-5` (both rounds), `MAX_DOCUMENT_LENGTH=384`, `POSITIVE_MIN_SCORE=0.2`, no prompt mapping during training, and a warmup bug that effectively disabled the learning rate warmup schedule.

Evaluation used 184 scene-based queries from 62 held-out movies:

| Model | MRR | Acc@5 | Acc@10 | Acc@20 | Median Rank |
|---|---|---|---|---|---|
| **Base** (no fine-tuning) | **0.421** | 0.511 | 0.592 | 0.669 | 2 |
| **Round 1** | 0.376 | 0.484 | 0.582 | 0.685 | 3 |
| **Round 2** | 0.000 | 0.000 | 0.000 | 0.000 | — |

Key observations:
- **Round 1 degraded** precision relative to the base model (MRR dropped from 0.421 to 0.376) despite a small recall gain at Acc@20.
- **Round 2 collapsed completely** — all zero scores indicate total embedding collapse, where every embedding converges to nearly the same vector and similarity scores become uninformative.

### Issues Identified

Six root causes were identified through systematic diagnosis:

**1. Warmup schedule bug** (`trainer.py`).  The `warmup_ratio` parameter (float 0.1) was assigned to `warmup_steps` (expects an integer number of steps).  This effectively set warmup to 0 steps, meaning the full learning rate was applied from step 1 — causing unstable early gradients.

**2. Train/eval prompt mismatch** (`trainer.py`).  EmbeddingGemma uses task-specific prompt prefixes (`"task: search result | query: "` for queries, `"title: none | text: "` for documents) during inference.  However, the `SentenceTransformerTrainingArguments` had no `prompts` mapping, so during training all text was encoded without any prefix.  The model learned representations in "unprompted" space but was evaluated in "prompted" space — a domain mismatch.

**3. Same learning rate for Round 2**.  Both rounds used `LEARNING_RATE=2e-5`.  Round 2 starts from a partially-optimised model, so the same aggressive rate causes catastrophic forgetting — the model unlearns the domain signal from Round 1 while adapting to the hard-negative distribution.

**4. Document truncation at 384 tokens** (`config.py`).  With `MAX_DOCUMENT_LENGTH=384`, scenes were truncated to ~300 words.  The average positive scene is ~850 words, meaning more than half the content was discarded.  When the relevant information falls beyond the truncation point, the positive becomes noise.

**5. Low positive threshold** (`config.py`).  `POSITIVE_MIN_SCORE=0.2` allowed low-confidence query-scene assignments for Type A (synopsis) queries.  These noisy positives dilute the training signal — the model is told "this query matches this scene" when the base model itself was not confident in the match.

**6. Hard-negative mining without prompts** (`negative_miner.py`).  `CorpusIndex.build()` and `mine_hard_negatives()` used bare `model.encode()` with empty prefix strings instead of the model's native `encode_query()` / `encode_document()` methods.  This meant mining operated in a different embedding space than evaluation, producing suboptimal hard negatives.

An additional issue was discovered during evaluation debugging: **FP16 overflow**.  The model was loaded in `float16` (via `FINETUNING_EMBED_FP16=true`), but the internal Dense projection layers overflow in half-precision, producing `NaN` embeddings.  This was fixed by switching to `bfloat16` (wider dynamic range) and casting outputs to `float32`.

### Fixes Applied (v2)

All fixes live on the `feat/finetune-v2` branch:

| Fix | File | Change |
|---|---|---|
| Warmup bug | `trainer.py` | `warmup_steps=warmup_ratio` → `warmup_ratio=warmup_ratio` |
| Prompt mapping | `trainer.py` | Added `prompts={"anchor": "query", "positive": "document", ...}` to training args |
| Round 2 LR | `config.py` | Added `ROUND2_LEARNING_RATE = 5e-6` (4x lower than Round 1) |
| Document length | `config.py` | `MAX_DOCUMENT_LENGTH` 384 → 512 tokens (~33% more context per scene) |
| Positive threshold | `config.py` | `POSITIVE_MIN_SCORE` 0.2 → 0.35 (filters noisy low-confidence matches) |
| Mining prompts | `negative_miner.py` | Uses `encode_queries()` / `encode_documents()` instead of bare `model.encode()` |
| FP16 → BF16 | `model.py` | `torch.float16` → `torch.bfloat16`; output cast to `float32` |

These are code-only changes — retraining is required to produce new model weights that benefit from the fixes.

---

## 11. Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | API key for OpenRouter (required for dataset generation) |
| `FEDE_LLM_MODEL` | Override the LLM model used for query generation (default: `google/gemini-2.0-flash-lite`) |
| `QDRANT_*` | All Qdrant settings — see `docs/vector_store.md` for the full list |

Importing `finetuning.config` loads the project-root **`.env`** via `python-dotenv`, so you can keep `OPENROUTER_API_KEY` in `.env` without exporting it in the shell. You can still override with `export` or `--api-key`.

---

## 12. CLI Reference

All scripts are invoked as Python modules from the project root:

```bash
# Build round-1 training dataset (LLM query generation)
python -m finetuning.scripts.build_dataset --movies 1200

# Clean raw queries (repair, score, deduplicate)
python -m finetuning.scripts.clean_raw_queries

# Generate scene-based eval dataset
python -m finetuning.scripts.generate_eval --scenes-per-movie 3

# Train round 1 (with per-epoch eval)
python -m finetuning.scripts.train \
    --round 1 \
    --output fede-embeddinggemma/round1 \
    --eval-dataset data/finetuning/eval_queries.json

# Mine hard negatives using round-1 model
python -m finetuning.scripts.mine_hard_negatives \
    --model fede-embeddinggemma/round1

# Train round 2 (with hard negatives, lower LR)
python -m finetuning.scripts.train \
    --round 2 \
    --model fede-embeddinggemma/round1 \
    --output fede-embeddinggemma/round2 \
    --eval-dataset data/finetuning/eval_queries.json \
    --lr 5e-6

# Index corpus into Qdrant with fine-tuned model
python -m finetuning.scripts.index_corpus \
    --model fede-embeddinggemma/round2
```

### Expected Data Volumes

For 1,200 movies:

| Data | Approximate count |
|---|---|
| Type A queries (raw) | ~4,800 |
| Type B queries (raw) | ~3,600 |
| Type C paraphrases (raw) | ~1,700 |
| After cleaning + dedup | ~8,000–9,000 |
| **Round 1 training pairs** | **~7,500 pairs** (after POSITIVE_MIN_SCORE filtering) |
| Round 2 training pairs | same pairs with hard negatives replacing random |
| Eval set | ~184 queries from ~62 held-out movies |
