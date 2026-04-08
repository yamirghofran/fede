"""FastAPI backend for semantic script retrieval."""

from .app import app, create_app
from .embedder import QueryEmbedder
from .runtime import BackendRuntime
from .search import MovieSearchResult, SemanticSearchService
from .settings import BackendSettings

__all__ = [
    "app",
    "create_app",
    "BackendRuntime",
    "BackendSettings",
    "MovieSearchResult",
    "QueryEmbedder",
    "SemanticSearchService",
]
