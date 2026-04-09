"""Backend runtime container for the FastAPI app."""

from __future__ import annotations

import threading
from typing import Optional

from knowledge_graph.graph_store import KnowledgeGraphService
from vector_db.client import get_client_info, reset_client
from vector_db.collections import CollectionManager
from vector_db.config import QdrantConfig
from vector_db.retrieval import ScriptRetriever
from vector_db.schemas import CollectionNames

from .embedder import QueryEmbedder
from .search import SemanticSearchService
from .settings import BackendSettings


class BackendRuntime:
    """Owns the long-lived backend services."""

    def __init__(
        self,
        settings: Optional[BackendSettings] = None,
        search_service: Optional[SemanticSearchService] = None,
        graph_service: Optional[KnowledgeGraphService] = None,
    ):
        self.settings = settings or BackendSettings()
        self.search_service = search_service
        self.graph_service = graph_service
        self.qdrant_config: Optional[QdrantConfig] = None
        self.startup_error: Optional[Exception] = None
        self.graph_startup_error: Optional[Exception] = None
        self.collection_status = {
            CollectionNames.SCENES.value: False,
            CollectionNames.SENTENCES.value: False,
        }
        self._lock = threading.Lock()

    def initialize(self) -> None:
        if self.graph_service is None:
            try:
                self.graph_service = KnowledgeGraphService(
                    db_path=self.settings.graph_db_path,
                    entities_dir=self.settings.graph_entities_dir,
                    relations_dir=self.settings.graph_relations_dir,
                )
                self.graph_service.initialize()
                self.graph_startup_error = None if self.graph_service.is_ready else RuntimeError("Grafeo backend is not available")
            except Exception as exc:
                self.graph_service = None
                self.graph_startup_error = exc
        if self.search_service is not None:
            self.startup_error = None
            return

        with self._lock:
            if self.search_service is not None:
                self.startup_error = None
                return

            try:
                qdrant_config = QdrantConfig.from_env()
                qdrant_config.validate()
                manager = CollectionManager(qdrant_config)
                self.collection_status = {
                    CollectionNames.SCENES.value: manager.collection_exists(CollectionNames.SCENES),
                    CollectionNames.SENTENCES.value: manager.collection_exists(CollectionNames.SENTENCES),
                }
                missing = [name for name, exists in self.collection_status.items() if not exists]
                if missing:
                    joined = ", ".join(sorted(missing))
                    raise ValueError(f"Required Qdrant collections are missing: {joined}")

                embedder = QueryEmbedder(
                    settings=self.settings,
                    vector_size=qdrant_config.vector_size,
                )
                embedder.load()
                retriever = ScriptRetriever(qdrant_config)

                self.qdrant_config = qdrant_config
                self.search_service = SemanticSearchService(
                    retriever=retriever,
                    embedder=embedder,
                    settings=self.settings,
                )
                self.startup_error = None
            except Exception as exc:
                self.qdrant_config = None
                self.search_service = None
                self.startup_error = exc

    def shutdown(self) -> None:
        reset_client()

    @property
    def is_ready(self) -> bool:
        return self.search_service is not None and self.startup_error is None

    def ensure_ready(self) -> SemanticSearchService:
        self.initialize()
        if self.search_service is None or self.startup_error is not None:
            raise RuntimeError(self.error_message)
        return self.search_service

    def ensure_graph_ready(self) -> KnowledgeGraphService:
        self.initialize()
        if self.graph_service is None:
            message = "Knowledge graph backend is not ready"
            if self.graph_startup_error is not None:
                message = str(self.graph_startup_error)
            raise RuntimeError(message)
        return self.graph_service

    @property
    def error_message(self) -> str:
        if self.startup_error is None:
            return "Backend is not ready"
        return str(self.startup_error)

    def readiness(self) -> dict:
        self.initialize()
        model_info = (
            self.search_service.embedder.info()
            if self.search_service is not None
            else {
                "loaded": False,
                "model_id": self.settings.embedding_model_id,
            }
        )
        return {
            "status": "ready" if self.is_ready else "not_ready",
            "collections": self.collection_status,
            "model": model_info,
            "qdrant": get_client_info(),
            "error": None if self.startup_error is None else str(self.startup_error),
        }
