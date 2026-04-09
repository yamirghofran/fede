"""Semantic search service used by the FastAPI backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from vector_db.retrieval import SceneResult, ScriptRetriever

from .embedder import QueryEmbedder
from .settings import BackendSettings


@dataclass
class MovieSearchResult:
    """Best matching scene representative for one movie."""

    movie_id: str
    movie_title: str
    score: float
    best_scene: SceneResult


class SemanticSearchService:
    """Encodes raw queries and searches the semantic store."""

    def __init__(
        self,
        retriever: ScriptRetriever,
        embedder: QueryEmbedder,
        settings: BackendSettings,
    ):
        self._retriever = retriever
        self._embedder = embedder
        self._settings = settings

    @property
    def embedder(self) -> QueryEmbedder:
        return self._embedder

    def search_movies(
        self,
        query: str,
        top_k: int | None = None,
        sentence_pool: int | None = None,
    ) -> List[MovieSearchResult]:
        resolved_top_k = self._resolve_top_k(top_k)
        resolved_sentence_pool = self._resolve_sentence_pool(sentence_pool)
        query_embedding = self._embedder.encode_query(query)

        scene_hits = self._retriever.hierarchical_search(
            query_embedding=query_embedding,
            top_k=max(resolved_top_k * self._settings.movie_overfetch_factor, resolved_top_k),
            sentence_pool=resolved_sentence_pool,
        )

        by_movie: dict[str, MovieSearchResult] = {}
        for scene in scene_hits:
            current = by_movie.get(scene.movie_id)
            if current is None or scene.score > current.score:
                by_movie[scene.movie_id] = MovieSearchResult(
                    movie_id=scene.movie_id,
                    movie_title=scene.movie_title,
                    score=scene.score,
                    best_scene=scene,
                )

        ranked = sorted(by_movie.values(), key=lambda hit: hit.score, reverse=True)
        return ranked[:resolved_top_k]

    def search_scenes(
        self,
        query: str,
        top_k: int | None = None,
        sentence_pool: int | None = None,
    ) -> List[SceneResult]:
        query_embedding = self._embedder.encode_query(query)
        return self._retriever.hierarchical_search(
            query_embedding=query_embedding,
            top_k=self._resolve_top_k(top_k),
            sentence_pool=self._resolve_sentence_pool(sentence_pool),
        )

    def _resolve_top_k(self, top_k: int | None) -> int:
        value = self._settings.default_top_k if top_k is None else top_k
        if value < 1:
            raise ValueError("top_k must be >= 1")
        if value > self._settings.max_top_k:
            raise ValueError(f"top_k must be <= {self._settings.max_top_k}")
        return value

    def _resolve_sentence_pool(self, sentence_pool: int | None) -> int:
        value = (
            self._settings.default_sentence_pool
            if sentence_pool is None
            else sentence_pool
        )
        if value < 1:
            raise ValueError("sentence_pool must be >= 1")
        if value > self._settings.max_sentence_pool:
            raise ValueError(
                f"sentence_pool must be <= {self._settings.max_sentence_pool}"
            )
        return value
