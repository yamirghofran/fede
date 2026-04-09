"""Offline script indexer — writes scene and sentence vectors into Qdrant."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

from .client import get_qdrant_client
from .collections import CollectionManager
from .config import QdrantConfig
from .crud import BaseVectorCRUD
from qdrant_client.models import PointStruct
from .schemas import (
    CollectionNames,
    LineType,
    ScenePayload,
    SentencePayload,
)

log = logging.getLogger(__name__)

# Stable namespace so deterministic UUIDs are consistent across runs and
# environments regardless of Python's hash seed.
_FEDE_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _scene_point_id(movie_id: str, scene_id: str) -> str:
    """Deterministic UUID for a scene vector — stable across re-indexing runs."""
    return str(uuid.uuid5(_FEDE_NS, f"{movie_id}::{scene_id}"))


def _sentence_point_id(movie_id: str, scene_id: str, position_in_script: int) -> str:
    """Deterministic UUID for a sentence vector."""
    return str(uuid.uuid5(_FEDE_NS, f"{movie_id}::{scene_id}::{position_in_script}"))


@dataclass
class SceneRecord:
    """Input record for a single scene to be indexed."""

    movie_id: str
    movie_title: str
    scene_id: str
    scene_index: int
    text: str
    embedding: List[float]
    scene_title: Optional[str] = None
    character_names: Optional[List[str]] = None


@dataclass
class SentenceRecord:
    """Input record for a single sentence / line to be indexed."""

    movie_id: str
    movie_title: str
    scene_id: str
    scene_index: int
    text: str
    line_type: LineType
    position_in_script: int
    embedding: List[float]
    character_name: Optional[str] = None


def _upsert_with_split(
    client,
    collection_name: str,
    points: List[PointStruct],
    wait: bool,
    _depth: int = 0,
) -> None:
    """Upsert points, splitting the batch in half on payload-too-large errors.

    Qdrant caps individual HTTP request bodies at 32 MB.  Very long scripts can
    exceed this with a single-movie batch.  Splitting is safe because every
    point has a deterministic UUID: the two sub-batches contain *disjoint* IDs,
    so the second POST never touches the points written by the first.

    Recursion stops at depth 4 (≤ 16 sub-batches).  If a single-point payload
    is still too large (theoretically impossible with our data) the exception
    propagates to the caller.
    """
    if not points:
        return
    try:
        client.upsert(collection_name=collection_name, points=points, wait=wait)
    except Exception as exc:
        if "larger than allowed" in str(exc) and len(points) > 1 and _depth < 4:
            mid = len(points) // 2
            log.warning(
                "Payload too large (%d points) — splitting into two halves (depth %d).",
                len(points),
                _depth + 1,
            )
            _upsert_with_split(client, collection_name, points[:mid], wait, _depth + 1)
            _upsert_with_split(client, collection_name, points[mid:], wait, _depth + 1)
        else:
            raise


class ScriptIndexer:
    """Writes preprocessed script segments into the Qdrant scenes and sentences
    collections.

    The indexer is intentionally decoupled from the embedding model — callers
    are responsible for supplying pre-computed embedding vectors. This keeps the
    indexer reusable across the baseline (un-fine-tuned) and fine-tuned variants
    of EmbeddingGemma without any code changes here.

    All write operations use upsert semantics with deterministic point IDs, so
    the indexer is safe to re-run: re-indexing the same movie overwrites existing
    vectors rather than creating duplicates.
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        self.config = config or QdrantConfig.from_env()
        client = get_qdrant_client(self.config)
        self._scenes = BaseVectorCRUD(
            collection=CollectionNames.SCENES.value,
            client=client,
            vector_size=self.config.vector_size,
        )
        self._sentences = BaseVectorCRUD(
            collection=CollectionNames.SENTENCES.value,
            client=client,
            vector_size=self.config.vector_size,
        )

    # ------------------------------------------------------------------
    # Single-record methods
    # ------------------------------------------------------------------

    def index_scene(self, record: SceneRecord) -> str:
        """Upsert a single scene vector. Returns the point ID."""
        point_id = _scene_point_id(record.movie_id, record.scene_id)
        payload: ScenePayload = {
            "movie_id": record.movie_id,
            "movie_title": record.movie_title,
            "scene_id": record.scene_id,
            "scene_index": record.scene_index,
            "text": record.text,
            "scene_title": record.scene_title,
            "character_names": record.character_names or [],
        }
        self._upsert_point(self._scenes, point_id, record.embedding, payload)
        return point_id

    def index_sentence(self, record: SentenceRecord) -> str:
        """Upsert a single sentence vector. Returns the point ID."""
        point_id = _sentence_point_id(
            record.movie_id, record.scene_id, record.position_in_script
        )
        payload: SentencePayload = {
            "movie_id": record.movie_id,
            "movie_title": record.movie_title,
            "scene_id": record.scene_id,
            "scene_index": record.scene_index,
            "text": record.text,
            "line_type": record.line_type,
            "character_name": record.character_name,
            "position_in_script": record.position_in_script,
        }
        self._upsert_point(self._sentences, point_id, record.embedding, payload)
        return point_id

    # ------------------------------------------------------------------
    # Batch method — primary entry point for the offline pipeline
    # ------------------------------------------------------------------

    def index_movie_batch(
        self,
        scenes: List[SceneRecord],
        sentences: List[SentenceRecord],
    ) -> None:
        """Upsert all scenes and sentences for one movie in two bulk calls.

        This is the main entry point during the offline indexing stage. Both
        calls use Qdrant's native batch upsert, which is significantly faster
        than individual inserts for large scripts.

        Args:
            scenes: All scene records for the movie, in script order.
            sentences: All sentence records for the movie, in script order.
        """
        if scenes:
            self._batch_upsert_scenes(scenes)
        if sentences:
            self._batch_upsert_sentences(sentences)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _upsert_point(
        crud: BaseVectorCRUD,
        point_id: str,
        embedding: List[float],
        payload: dict,
    ) -> None:
        """Upsert a single point, bypassing the duplicate-check in crud.add()."""

        crud.client.upsert(
            collection_name=crud.collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
            wait=True,
        )

    def _batch_upsert_scenes(self, records: List[SceneRecord]) -> None:

        points = [
            PointStruct(
                id=_scene_point_id(r.movie_id, r.scene_id),
                vector=r.embedding,
                payload=ScenePayload(
                    movie_id=r.movie_id,
                    movie_title=r.movie_title,
                    scene_id=r.scene_id,
                    scene_index=r.scene_index,
                    text=r.text,
                    scene_title=r.scene_title,
                    character_names=r.character_names or [],
                ),
            )
            for r in records
        ]
        _upsert_with_split(self._scenes.client, self._scenes.collection_name, points, wait=False)

    def _batch_upsert_sentences(self, records: List[SentenceRecord]) -> None:
        points = [
            PointStruct(
                id=_sentence_point_id(r.movie_id, r.scene_id, r.position_in_script),
                vector=r.embedding,
                payload=SentencePayload(
                    movie_id=r.movie_id,
                    movie_title=r.movie_title,
                    scene_id=r.scene_id,
                    scene_index=r.scene_index,
                    text=r.text,
                    line_type=r.line_type,
                    character_name=r.character_name,
                    position_in_script=r.position_in_script,
                ),
            )
            for r in records
        ]
        _upsert_with_split(self._sentences.client, self._sentences.collection_name, points, wait=False)


def index_movie(
    scenes: List[SceneRecord],
    sentences: List[SentenceRecord],
    config: Optional[QdrantConfig] = None,
) -> None:
    """Module-level convenience wrapper around ScriptIndexer.index_movie_batch."""
    ScriptIndexer(config).index_movie_batch(scenes, sentences)
