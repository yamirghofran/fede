# FEDE — Film Embedding and Discovery Engine

Semantic and structural search over movie scripts. FEDE combines fine-tuned sentence embeddings with a narrative knowledge graph to retrieve scenes based on story patterns, not just keyword matches.

## Overview

FEDE indexes movie scripts at two granularities: scene-level vectors capture broad context, while sentence-level vectors preserve specific dialogue and action details. A knowledge graph built from extracted entities and relations enables motif-based queries (e.g., "mentor betrays student"). The hybrid retrieval mode combines both signals, ranking results by semantic similarity and narrative structure.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  React UI   │────▶│  FastAPI     │────▶│  Qdrant Vector  │
│  (apps/web) │◄────│  (apps/api)  │◄────│  Store          │
└─────────────┘     └──────────────┘     └─────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Grafeo KG   │
                     │  (in-process)│
                     └──────────────┘
```

**Vector Store**: Qdrant hosts two collections—`scenes` and `sentences`—with 768-dimensional vectors from a fine-tuned EmbeddingGemma model. Hierarchical search first retrieves candidate sentences, then resolves parent scenes and merges scores with a direct scene search.

**Knowledge Graph**: Built using Grafeo, the graph stores entities (characters, events, concepts) and relations (BETRAYS, TEACHES, SAVES). Pattern queries traverse the graph to find movies matching specific narrative structures.

**Embedding Model**: Fine-tuned on synthetic queries generated from movie synopses and scene summaries. Uses LoRA adapters for efficient training on consumer GPUs.

## Quick Start

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your HF_TOKEN, GEMINI_API_KEY, etc.

# Start Qdrant (requires Docker)
docker-compose up -d

# Run the API
python -m apps.api
```

## Project Structure

```
apps/
  api/              # FastAPI backend
    app.py          # Route definitions
    embedder.py     # HuggingFace model loading
    search.py       # Semantic search service
    hybrid.py       # LLM translation + hybrid ranking
    runtime.py      # Service lifecycle management
  web/              # React frontend
    src/routes/index.tsx  # Query console UI

vector_db/          # Qdrant client and indexing
  client.py         # Singleton Qdrant client
  indexer.py        # Script → vectors
  retrieval.py      # Hierarchical search
  schemas.py        # Typed payload definitions

knowledge_graph/    # Graph storage and queries
  graph_store.py    # Grafeo backend wrapper
  graph_models.py   # Pydantic models
  predicates.py     # Valid relation types
  relation_extraction.py  # LLM-based extraction

preprocessing/      # Script parsing
  chunker.py        # Tagged format → chunks

finetuning/         # Model training
  dataset/          # Synthetic query generation
  training/         # LoRA fine-tuning
  evaluation/       # Retrieval metrics

data/               # Indexed data (gitignored)
  scripts/          # Raw and parsed scripts
  grafeo/           # Graph database
```

## Key Features

**Three Retrieval Modes**:
- **Semantic**: Direct vector similarity for implicit relations in tone or dialogue
- **Knowledge Graph**: Motif search using narrative predicates (TEACHES, BETRAYS, etc.)
- **Hybrid**: Combines both signals, ranking by combined semantic and graph scores

**Query Translation**: Natural language queries are translated to graph patterns using an LLM, falling back to semantic-only when no clear motif is detected.

**Hierarchical Search**: Two-stage retrieval—sentence search for granularity, scene search for context—merged via max-pooling.

## Configuration

Key environment variables (see `.env.example`):

| Variable | Purpose | Default |
|----------|---------|---------|
| `EMBEDDING_MODEL_ID` | HuggingFace model identifier | `google/embeddinggemma-300m` |
| `QDRANT_MODE` | Connection mode (local/server) | `server` |
| `QDRANT_HOST` | Qdrant server hostname | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `GRAPH_DB_PATH` | Path to Grafeo database | `./data/grafeo/story_graph.db` |
| `LLM_API_KEY` | API key for query translation | — |
| `LLM_MODEL` | Model for pattern translation | — |

## Deployment

Deployed on Railway with separate services for the FastAPI backend and Qdrant vector store. The knowledge graph runs in-process to eliminate network overhead. Current infrastructure handles ~100,000 scripts with vertical scaling; horizontal sharding would be needed beyond that.

Measured latency: 73ms backend response, 127ms Qdrant search.

## Development

```bash
# Run tests
pytest

# Format code
ruff check --fix .
black .

# Type checking (optional)
mypy apps/api vector_db knowledge_graph
```

## License

MIT
