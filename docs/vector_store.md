# FEDE — Film Embedding and Discovery Engine

FEDE is a retrieval engine for full movie scripts. It enables semantic search over narrative content by combining a vector-based semantic retrieval module with a knowledge graph-based structural retrieval module. The platform is designed to help writers and producers understand narrative composition and discover structurally or thematically similar scripts.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Quick Start](#3-quick-start)
4. [Configuration](#4-configuration)
5. [Vector DB Module](#5-vector-db-module)
   - [Collections](#collections)
   - [Schemas and Payloads](#schemas-and-payloads)
   - [Initialising Collections](#initialising-collections)
   - [Connecting from the Preprocessing Pipeline](#connecting-from-the-preprocessing-pipeline)
   - [Connecting from the RAG / Online Pipeline](#connecting-from-the-rag--online-pipeline)
6. [Running Tests](#6-running-tests)
7. [Notebook Smoke Test](#7-notebook-smoke-test)
8. [Environment Variables Reference](#8-environment-variables-reference)

---

## 1. Architecture Overview

FEDE operates in two distinct phases:

```
╔══════════════════════════════════════════════════════════════╗
║  OFFLINE  (run once per dataset update)                      ║
║                                                              ║
║  Raw Scripts ──► Preprocessor ──► EmbeddingGemma ──► Qdrant ║
║                         │                                    ║
║                         └──► LLM (Gemini) ──► Neo4j         ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  ONLINE  (per user query)                                    ║
║                                                              ║
║  User Query ──► QueryEnricher ──► EmbeddingGemma             ║
║                                          │                   ║
║                              ┌───────────┴────────────┐     ║
║                              ▼                         ▼     ║
║                    Semantic Retrieval        Structural       ║
║                      (Qdrant)               Retrieval        ║
║                              │             (Neo4j)           ║
║                              └───────────┬────────────┘     ║
║                                          ▼                   ║
║                                 Hybrid Aggregation           ║
║                                  S_Hybrid = S_SRM + λ·S_StRM ║
╚══════════════════════════════════════════════════════════════╝
```

The `vector_db/` package covers the **Qdrant side** of both phases: offline indexing and online semantic retrieval.

---

## 2. Repository Structure

```
fede/
├── vector_db/              # Qdrant semantic retrieval module
│   ├── __init__.py         # Public API surface
│   ├── config.py           # QdrantConfig (env var loading + validation)
│   ├── client.py           # Qdrant client singleton
│   ├── collections.py      # CollectionManager (scenes + sentences)
│   ├── schemas.py          # CollectionNames enum, SentencePayload, ScenePayload
│   ├── crud.py             # Generic BaseVectorCRUD
│   ├── indexer.py          # ScriptIndexer — offline batch upsert
│   └── retrieval.py        # ScriptRetriever — online hierarchical search
├── tests/
│   ├── test_config.py
│   ├── test_schemas.py
│   ├── test_indexer.py
│   └── test_retrieval.py
├── notebooks/
│   └── vector_db_smoke_test.ipynb   # End-to-end smoke test against live Qdrant
├── docker-compose.yml      # Qdrant container
├── .env.example            # Environment variable template
└── pyproject.toml          # Python project metadata and dependencies
```

---

## 3. Quick Start

### Prerequisites

- Docker Desktop
- Python ≥ 3.12 with a virtual environment

### 1. Start Qdrant

```bash
docker compose up -d
docker compose ps   # wait until Status shows "healthy"
```

### 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if you need non-default values (port, API key, etc.)
```

### 4. Initialise collections

```python
from vector_db import initialize_all_collections

manager = initialize_all_collections()
print(manager.list_collections())  # ['scenes', 'sentences']
```

---

## 4. Configuration

All runtime settings are read from environment variables. Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `QDRANT_MODE` | `server` | `server` (Docker/hosted) or `local` (embedded file) |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant REST port |
| `QDRANT_API_KEY` | _(unset)_ | API key — leave unset for local dev |
| `QDRANT_HTTPS` | `false` | Enable TLS for hosted deployments |
| `QDRANT_VECTOR_SIZE` | `768` | Must match your embedding model output dimension |
| `QDRANT_SCRIPTS_ON_DISK` | `true` | Store vectors on disk rather than RAM |
| `QDRANT_HNSW_ON_DISK` | `true` | Store HNSW graph on disk |
| `QDRANT_HNSW_M` | `16` | HNSW connectivity parameter |
| `QDRANT_INT8_QUANTIZATION` | `true` | Enable INT8 scalar quantization |
| `QDRANT_QUANTILE` | `0.99` | Quantization calibration quantile |
| `QDRANT_QUANT_ALWAYS_RAM` | `true` | Keep quantized vectors in RAM |
| `QDRANT_TIMEOUT` | `10.0` | Client timeout in seconds |

`QdrantConfig` can also be constructed programmatically for environments where `.env` is not used:

```python
from vector_db import QdrantConfig

cfg = QdrantConfig(
    mode="server",
    host="localhost",
    port=6333,
    vector_size=768,
    int8_quantization=False,
)
```

---

## 5. Vector DB Module

### Collections

FEDE maintains two Qdrant collections, both using cosine distance and sharing the same vector configuration:

| Collection | Purpose |
|---|---|
| `scenes` | One vector per scene; represents the aggregated semantic content of an entire scene (concatenated descriptions + dialogue). Primary retrieval unit returned to users. |
| `sentences` | One vector per line; captures granular dialogue and action detail. Used in stage 1 of hierarchical search to surface parent scenes that a broad scene-level search might miss. |

### Schemas and Payloads

Every indexed vector carries a typed payload alongside it.

**`SentencePayload`** — stored with each sentence vector:

| Field | Type | Description |
|---|---|---|
| `movie_id` | `str` | Unique movie identifier (e.g. TMDB ID) |
| `movie_title` | `str` | Human-readable title |
| `scene_id` | `str` | Parent scene identifier — links back to scenes collection |
| `scene_index` | `int` | Zero-based scene position in script |
| `text` | `str` | Raw line text |
| `line_type` | `"dialogue" \| "description" \| "transition"` | Content classification from preprocessing |
| `character_name` | `str \| None` | Speaker for dialogue lines; `None` otherwise |
| `position_in_script` | `int` | Absolute zero-based line index in the full script |

**`ScenePayload`** — stored with each scene vector:

| Field | Type | Description |
|---|---|---|
| `movie_id` | `str` | Unique movie identifier |
| `movie_title` | `str` | Human-readable title |
| `scene_id` | `str` | Unique scene identifier |
| `scene_index` | `int` | Zero-based scene position in script |
| `text` | `str` | Full scene text (descriptions + dialogue) |
| `scene_title` | `str \| None` | Slugline if present (e.g. `INT. COFFEE SHOP - DAY`) |
| `character_names` | `List[str]` | Deduplicated character list for structural filtering |

### Initialising Collections

```python
from vector_db import initialize_all_collections, CollectionNames

manager = initialize_all_collections()

# Check status
for col in CollectionNames:
    print(f"{col.value}: {manager.get_collection_count(col)} vectors")

# Reset a collection (wipes all indexed vectors)
manager.reset_collection(CollectionNames.SCENES)
```

---

### Connecting from the Preprocessing Pipeline

The preprocessing pipeline is responsible for:
1. Parsing raw scripts into tagged lines (`dialogue` / `description` / `transition`)
2. Grouping lines into scenes
3. Computing embeddings via EmbeddingGemma

Once those steps are complete, hand the results to `ScriptIndexer`:

```python
from vector_db import ScriptIndexer, SceneRecord, SentenceRecord, QdrantConfig

# Initialise — reads QDRANT_* env vars by default
indexer = ScriptIndexer()

# Build records from your preprocessor's output
scenes = [
    SceneRecord(
        movie_id="tt0111161",           # TMDB or internal ID
        movie_title="The Shawshank Redemption",
        scene_id="tt0111161-scene-000", # must be unique per movie
        scene_index=0,                  # position in script (0-based)
        text="INT. PRISON CELL - NIGHT\nAndy sits alone...",
        embedding=embedder.encode(scene_text),  # List[float], dim=768
        scene_title="INT. PRISON CELL - NIGHT",
        character_names=["ANDY", "GUARD"],
    ),
    # ... one SceneRecord per scene
]

sentences = [
    SentenceRecord(
        movie_id="tt0111161",
        movie_title="The Shawshank Redemption",
        scene_id="tt0111161-scene-000",  # must match parent SceneRecord
        scene_index=0,
        text="Get busy living, or get busy dying.",
        line_type="dialogue",            # "dialogue" | "description" | "transition"
        position_in_script=42,          # absolute line index in full script
        embedding=embedder.encode(line_text),
        character_name="RED",           # None for non-dialogue lines
    ),
    # ... one SentenceRecord per line
]

# Index an entire movie in two bulk upsert calls (idempotent — safe to re-run)
indexer.index_movie_batch(scenes, sentences)
```

**Key properties of the indexer:**
- **Idempotent**: point IDs are deterministic `uuid5` hashes of `movie_id::scene_id` (scenes) and `movie_id::scene_id::position` (sentences). Re-indexing the same movie silently overwrites — no duplicates are created.
- **Decoupled from EmbeddingGemma**: embeddings are passed in pre-computed. This means the indexer works with both the baseline and fine-tuned model variants without code changes.
- **Batch writes**: `index_movie_batch` issues a single bulk upsert per collection, which is significantly faster than per-record inserts for large scripts.

---

### Connecting from the RAG / Online Pipeline

The online pipeline receives a user query and must return a ranked list of movies or scenes. The `ScriptRetriever` is the entry point for the semantic retrieval branch.

#### Typical call sequence

```
User query (natural language)
    │
    ▼
QueryEnricher.enrich(query)          ← rewrites query into script style
    │
    ▼
EmbeddingGemma.encode(enriched)      ← produces a 768-dim vector
    │
    ▼
ScriptRetriever.hierarchical_search(embedding, top_k=20)
    │
    ▼
List[SceneResult]                    ← ranked by merged score
    │
    ▼
Hybrid aggregation                   ← combine with Neo4j structural results
    │
    ▼
Final ranking: S_Hybrid = S_SRM(m) + λ · S_StRM(m)
```

#### Using ScriptRetriever

```python
from vector_db import ScriptRetriever

retriever = ScriptRetriever()  # reads QDRANT_* env vars

# --- Hierarchical search (recommended for production) ---
# Stage 1: retrieve sentence_pool candidate sentences (granular recall)
# Stage 2: resolve to parent scenes, merge with direct scene search, re-rank
results = retriever.hierarchical_search(
    query_embedding=query_vector,   # List[float] from EmbeddingGemma
    top_k=20,                       # final number of scenes to return
    sentence_pool=100,              # sentences to fetch in stage 1 (recall/speed trade-off)
)

for scene in results:
    print(f"[{scene.score:.4f}] {scene.movie_title} — {scene.scene_title}")
    print(f"  Characters: {scene.character_names}")
    print(f"  {scene.text[:120]}...")

# --- Flat scene search (for ablation / baseline comparison) ---
scene_results = retriever.search_scenes(query_embedding=query_vector, top_k=20)

# --- Flat sentence search ---
sent_results = retriever.search_sentences(query_embedding=query_vector, top_k=50)

# --- Filter to a single movie (evaluation use case) ---
filtered = retriever.search_scenes(
    query_embedding=query_vector,
    top_k=5,
    movie_id_filter="tt0111161",
)
```

#### Result objects

`hierarchical_search` and `search_scenes` return `List[SceneResult]`:

```python
@dataclass
class SceneResult:
    point_id: str           # Qdrant point UUID
    score: float            # merged cosine similarity score (0–1)
    movie_id: str
    movie_title: str
    scene_id: str
    scene_index: int        # position in script — useful for temporal ordering
    text: str               # full scene text
    scene_title: str | None # slugline, e.g. "INT. COFFEE SHOP - DAY"
    character_names: list[str]
```

`search_sentences` returns `List[SentenceResult]`:

```python
@dataclass
class SentenceResult:
    point_id: str
    score: float
    movie_id: str
    movie_title: str
    scene_id: str           # use this to look up the parent scene
    scene_index: int
    text: str
    line_type: str          # "dialogue" | "description" | "transition"
    position_in_script: int
    character_name: str | None
```

#### Hybrid aggregation

The retriever returns the semantic score `S_SRM(m)`. To compute the hybrid score with the structural module:

```python
from vector_db import ScriptRetriever

retriever = ScriptRetriever()
semantic_results = retriever.hierarchical_search(query_embedding, top_k=50)

# Build a score map: movie_id → max scene score for that movie
semantic_scores: dict[str, float] = {}
for r in semantic_results:
    if r.movie_id not in semantic_scores or r.score > semantic_scores[r.movie_id]:
        semantic_scores[r.movie_id] = r.score

# structural_matches: set[str] of movie_ids from Neo4j query
lam = 0.5  # λ — controls structural contribution weight
hybrid_scores = {
    mid: semantic_scores.get(mid, 0.0) + lam * (1.0 if mid in structural_matches else 0.0)
    for mid in semantic_scores | {m: 0.0 for m in structural_matches}
}

ranked_movies = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)
```

---

## 6. Running Tests

```bash
# All tests (no live Qdrant required — all Qdrant calls are mocked)
.venv/bin/pytest tests/ -v

# A single module
.venv/bin/pytest tests/test_retrieval.py -v
```

The test suite covers 100 cases across config, schemas, indexer, and retrieval. All Qdrant calls are mocked — no running container is needed.

---

## 7. Notebook Smoke Test

`notebooks/vector_db_smoke_test.ipynb` exercises the full stack against a live Qdrant container using randomly-generated embeddings. Run it after any schema or retrieval logic change to confirm end-to-end correctness.

```bash
docker compose up -d
.venv/bin/jupyter lab
# Open notebooks/vector_db_smoke_test.ipynb and run all cells
```

The notebook covers:

| Section | What it verifies |
|---|---|
| Connection check | Qdrant is reachable with retry backoff |
| Collection init | Both collections created, counts are zero |
| Fake data generation | 2 movies × 5 scenes × 4 sentences |
| Indexing | `index_movie_batch` populates collections |
| Idempotency | Re-indexing does not create duplicates |
| Flat search — scenes | `search_scenes` returns correct count and movie_ids |
| Flat search — sentences | `search_sentences` returns correct `line_type` and `character_name` |
| `movie_id_filter` | Results restricted to one movie |
| Hierarchical search | Sorted, unique scene_ids, respects `top_k` |
| Sentence-only path | Exact sentence embedding surfaces parent scene |
| Deterministic IDs | `uuid5` stability across re-index calls |

---

## 8. Environment Variables Reference

See `.env.example` for a fully annotated template. The table below shows the recommended values for each deployment context:

| Variable | Local dev | Production |
|---|---|---|
| `QDRANT_MODE` | `server` | `server` |
| `QDRANT_HOST` | `localhost` | your cluster hostname |
| `QDRANT_PORT` | `6333` | `6333` |
| `QDRANT_API_KEY` | _(unset)_ | set a strong key |
| `QDRANT_HTTPS` | `false` | `true` |
| `QDRANT_VECTOR_SIZE` | `768` | `768` |
| `QDRANT_SCRIPTS_ON_DISK` | `false` | `true` |
| `QDRANT_HNSW_ON_DISK` | `false` | `true` |
| `QDRANT_INT8_QUANTIZATION` | `false` | `true` |
| `QDRANT_TIMEOUT` | `10.0` | `10.0` |
