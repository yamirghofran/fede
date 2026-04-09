## Deployment

The fine-tuned embedding model is published to HuggingFace Hub, where it can be versioned and accessed via the standard transformers API. At startup, the FastAPI backend service fetches this model and loads it into memory as part of its initialization routine. The model is pulled from HuggingFace using the `sentence-transformers` library with optional PEFT/LoRA adapter support, allowing the service to apply domain-specific fine-tuning on top of a base embedding model.

Railway serves as the platform-as-a-service provider for this deployment. The choice was driven by its straightforward Git-based deployment workflow, automatic environment variable management, and built-in support for both stateless services and containerized databases. Railway handles SSL termination, domain provisioning, and horizontal scaling without requiring custom infrastructure code.

Vector storage runs on a Qdrant image deployed as a separate service within the same Railway project. The backend connects to Qdrant via a URL and port combination exposed through environment variables (`QDRANT_URL` or `QDRANT_HOST`/`QDRANT_PORT`). The Qdrant instance maintains two collections—`scenes` and `sentences`—storing 768-dimensional vectors from the EmbeddingGemma model. INT8 scalar quantization is enabled in production to reduce memory footprint while preserving search quality.

The knowledge graph database operates differently from the vector store. Rather than deploying a separate graph database service, Grafeo runs in-process within the FastAPI application. The graph data is stored in a SQLite-compatible database file at a configurable path (`GRAPH_DB_PATH`). This design eliminates network overhead for graph traversals and simplifies deployment, though it limits the graph to single-node access.

Measured latency figures show 73ms for backend response time and 127ms for Qdrant vector search operations. These measurements were taken under typical load conditions with the embedding model loaded in memory and the graph database initialized.

The web interface provides three retrieval modes through a React-based console. Users can switch between semantic search (direct vector similarity), knowledge graph search (motif-based pattern matching), and hybrid search (combining both signals). The interface displays ranked results with movie metadata, scene excerpts, and when applicable, the graph translation showing extracted predicates like TEACHES, BETRAYS, or CONFRONTS.

Current infrastructure scales vertically to handle approximately 100,000 scripts. A single Railway instance with increased CPU and memory allocation can index and serve queries against this corpus. Horizontal sharding would be required to go beyond that threshold—partitioning the vector collections across multiple Qdrant nodes and distributing the graph database or replicating it across application instances.
