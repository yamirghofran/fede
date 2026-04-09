"""Pydantic models for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from knowledge_graph.graph_models import (
    GraphBuildRequest,
    GraphBuildResponse,
    GraphHealthResponse,
    MovieGraphResponse,
    PatternQueryRequest,
    PatternQueryResponse,
)


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    collections: Dict[str, bool]
    model: Dict[str, Any]
    qdrant: Dict[str, Any]
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=None, ge=1)
    sentence_pool: Optional[int] = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class SceneSnippet(BaseModel):
    point_id: str
    scene_id: str
    scene_index: int
    scene_title: Optional[str] = None
    score: float
    text: str
    character_names: list[str] = Field(default_factory=list)


class RankedSceneResult(SceneSnippet):
    rank: int
    movie_id: str
    movie_title: str


class RankedMovieResult(BaseModel):
    rank: int
    movie_id: str
    movie_title: str
    score: float
    best_scene: SceneSnippet


class MovieSearchResponse(BaseModel):
    query: str
    top_k: int
    sentence_pool: int
    results: list[RankedMovieResult]


class SceneSearchResponse(BaseModel):
    query: str
    top_k: int
    sentence_pool: int
    results: list[RankedSceneResult]


class HybridQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=None, ge=1)
    sentence_pool: Optional[int] = Field(default=None, ge=1)
    graph_limit: Optional[int] = Field(default=None, ge=1, le=50)
    use_semantic: bool = True
    use_graph: bool = True

    @field_validator("query")
    @classmethod
    def _validate_hybrid_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class GraphTranslationResponse(BaseModel):
    status: Literal["translated", "unavailable", "failed", "skipped"]
    pattern: Optional[PatternQueryRequest] = None
    error: Optional[str] = None


class HybridMovieResult(BaseModel):
    rank: int
    movie_id: str
    movie_title: str
    score: float
    semantic_score: Optional[float] = None
    graph_score: Optional[float] = None
    best_scene: Optional[SceneSnippet] = None
    graph_matches: list["PatternMatchSummary"] = Field(default_factory=list)


class PatternMatchSummary(BaseModel):
    score: float
    path_entities: list[str]
    predicates: list[str]
    evidences: list[str]


class HybridQueryResponse(BaseModel):
    query: str
    top_k: int
    sentence_pool: int
    graph_limit: int
    strategy: Literal["hybrid", "semantic_only", "graph_only", "no_results"]
    translation: GraphTranslationResponse
    results: list[HybridMovieResult]


HybridMovieResult.model_rebuild()
