"""FEDE vector_db package — Qdrant-backed semantic retrieval for movie scripts."""

from .client import get_qdrant_client, reset_client
from .collections import (
    CollectionManager,
    get_scenes_collection,
    get_sentences_collection,
    initialize_all_collections,
)
from .config import QdrantConfig
from .crud import BaseVectorCRUD
from .indexer import SceneRecord, ScriptIndexer, SentenceRecord, index_movie
from .schemas import CollectionNames, LineType, ScenePayload, SentencePayload

__all__ = [
    # config
    "QdrantConfig",
    # client
    "get_qdrant_client",
    "reset_client",
    # collections
    "CollectionManager",
    "initialize_all_collections",
    "get_scenes_collection",
    "get_sentences_collection",
    # schemas
    "CollectionNames",
    "LineType",
    "ScenePayload",
    "SentencePayload",
    # crud
    "BaseVectorCRUD",
    # indexer
    "ScriptIndexer",
    "SceneRecord",
    "SentenceRecord",
    "index_movie",
]
