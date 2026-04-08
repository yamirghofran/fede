# upload_scripts.py — Offline Indexing Pipeline

Reads tagged movie scripts, chunks them, embeds them, and bulk-upserts into Qdrant. Run this once per dataset update (or incrementally — it resumes automatically).

---

## What it does

```
Tagged scripts (data/scripts/parsed/tagged/)
    │
    ▼
ScriptChunker → scene chunks + sentence chunks
    │
    ▼
SentenceTransformer (EMBEDDING_MODEL_NAME) → float32 vectors
    │
    ▼
ScriptIndexer.index_movie_batch() → Qdrant (scenes + sentences collections)
    │
    ▼
.indexed_movies.txt  ← progress file; re-runs skip already-indexed movies
```

---

## Quick Start

```bash
# Single process (safest, shows tqdm progress bar)
python scripts/upload_scripts.py

# Limit to first 5 movies (smoke test)
python scripts/upload_scripts.py --limit 5

# 4 parallel workers on CPU
python scripts/upload_scripts.py --workers 4

# Re-index everything from scratch
python scripts/upload_scripts.py --no-resume
```

---

## Options

| Flag | Default | Description |
|---|---|---|
| `--batch-size N` | `EMBEDDING_BATCH_SIZE` env or `64` | Embedding batch size |
| `--device DEVICE` | `EMBEDDING_DEVICE` env or `auto` | `cpu` \| `cuda` \| `mps` \| `auto` |
| `--limit N` | _(none)_ | Process only the first N movies |
| `--no-resume` | `false` | Ignore progress file; re-index all |
| `--workers N` | `1` | Parallel worker processes (each loads its own model) |

---

## Required environment variables

| Variable | Example | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace model ID |
| `QDRANT_VECTOR_SIZE` | `768` | Must match model output dimension |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant REST port |

See `.env.example` and `docs/vector_store.md §4` for the full list.

---

## Resume behaviour

Progress is tracked in `scripts/.indexed_movies.txt` (one `movie_id` per line). On restart, already-indexed movies are skipped. Use `--no-resume` to force a full re-index.

Upserts are **idempotent** — re-indexing the same movie overwrites existing points without creating duplicates (IDs are deterministic `uuid5` hashes).

---

## Parallelism notes

- `--workers 1` (default): single process, tqdm progress bar, simplest to debug.
- `--workers N`: spawns N processes via `multiprocessing.Pool`; each loads its own model copy. Recommended: **2–4 on CPU** with ≥24 GB RAM.
- PyTorch intra-op threads are automatically capped per worker to avoid CPU over-subscription.
