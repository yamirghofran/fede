"""Domain and API models for the knowledge graph."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .predicates import VALID_ENTITY_TYPES, VALID_PREDICATES


class GraphEntity(BaseModel):
    entity_id: str
    movie_id: str
    canonical_name: str
    entity_type: str
    source_file: str

    @field_validator("entity_type")
    @classmethod
    def _validate_entity_type(cls, value: str) -> str:
        if value not in VALID_ENTITY_TYPES:
            raise ValueError(f"Unsupported entity type: {value}")
        return value


class NormalizedRelation(BaseModel):
    relation_id: str
    movie_id: str
    from_entity_id: str
    to_entity_id: str
    from_name: str
    from_type: str
    to_name: str
    to_type: str
    predicate: str
    evidence: str
    source_file: str

    @field_validator("predicate")
    @classmethod
    def _validate_predicate(cls, value: str) -> str:
        if value not in VALID_PREDICATES:
            raise ValueError(f"Unsupported predicate: {value}")
        return value


class DroppedRelation(BaseModel):
    movie_id: str
    reason: str
    raw_relation: dict


class MovieGraphDocument(BaseModel):
    movie_id: str
    title: str
    source_file: str
    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[NormalizedRelation] = Field(default_factory=list)
    dropped_relations: list[DroppedRelation] = Field(default_factory=list)


class BuildSummary(BaseModel):
    mode: Literal["full_rebuild", "movie_reload"]
    requested_movie_id: Optional[str] = None
    movies_loaded: int
    nodes_created: int
    edges_created: int
    dropped_relations: int
    dropped_by_reason: dict[str, int] = Field(default_factory=dict)
    db_path: str

    @classmethod
    def from_documents(
        cls,
        *,
        mode: Literal["full_rebuild", "movie_reload"],
        requested_movie_id: Optional[str],
        documents: list[MovieGraphDocument],
        db_path: Path,
    ) -> "BuildSummary":
        dropped_counter = Counter()
        for document in documents:
            dropped_counter.update(item.reason for item in document.dropped_relations)
        nodes_created = sum(1 + len(document.entities) for document in documents)
        edges_created = sum(len(document.entities) + len(document.relations) for document in documents)
        return cls(
            mode=mode,
            requested_movie_id=requested_movie_id,
            movies_loaded=len(documents),
            nodes_created=nodes_created,
            edges_created=edges_created,
            dropped_relations=sum(dropped_counter.values()),
            dropped_by_reason=dict(sorted(dropped_counter.items())),
            db_path=str(db_path),
        )


class GraphBuildRequest(BaseModel):
    movie_id: Optional[str] = None
    rebuild: bool = False

    @field_validator("movie_id")
    @classmethod
    def _normalize_movie_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class GraphBuildResponse(BuildSummary):
    pass


class GraphCounts(BaseModel):
    movies: int = 0
    entities: int = 0
    narrative_edges: int = 0
    total_edges: int = 0


class GraphHealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    db_path: str
    grafeo_available: bool
    counts: GraphCounts
    last_build: Optional[BuildSummary] = None
    error: Optional[str] = None


class MovieEntityResponse(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str


class MovieRelationResponse(BaseModel):
    relation_id: str
    predicate: str
    from_entity_id: str
    from_name: str
    to_entity_id: str
    to_name: str
    evidence: str


class MovieGraphResponse(BaseModel):
    movie_id: str
    title: str
    source_file: str
    entities: list[MovieEntityResponse]
    outgoing_relations: list[MovieRelationResponse]
    incoming_relations: list[MovieRelationResponse]


class PatternQueryRequest(BaseModel):
    predicates: list[str] = Field(min_length=1)
    entity_types: Optional[list[str]] = None
    movie_ids: Optional[list[str]] = None
    contains_entities: Optional[list[str]] = None
    limit: int = Field(default=10, ge=1, le=100)

    @field_validator("predicates")
    @classmethod
    def _validate_predicates(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip().upper() for value in values if value and value.strip()]
        if not cleaned:
            raise ValueError("predicates must not be empty")
        invalid = [value for value in cleaned if value not in VALID_PREDICATES]
        if invalid:
            joined = ", ".join(sorted(set(invalid)))
            raise ValueError(f"Unsupported predicates: {joined}")
        return cleaned

    @field_validator("entity_types")
    @classmethod
    def _validate_entity_types(cls, values: Optional[list[str]]) -> Optional[list[str]]:
        if values is None:
            return None
        cleaned = [value.strip().upper() for value in values if value and value.strip()]
        invalid = [value for value in cleaned if value not in VALID_ENTITY_TYPES]
        if invalid:
            joined = ", ".join(sorted(set(invalid)))
            raise ValueError(f"Unsupported entity types: {joined}")
        return cleaned or None

    @field_validator("movie_ids", "contains_entities")
    @classmethod
    def _strip_list(cls, values: Optional[list[str]]) -> Optional[list[str]]:
        if values is None:
            return None
        cleaned = [value.strip() for value in values if value and value.strip()]
        return cleaned or None


class PatternStep(BaseModel):
    entity_id: str
    entity_name: str
    entity_type: str


class PatternEdgeEvidence(BaseModel):
    relation_id: str
    predicate: str
    evidence: str


class PatternMatchResponse(BaseModel):
    movie_id: str
    movie_title: str
    score: float
    path: list[PatternStep]
    evidences: list[PatternEdgeEvidence]


class PatternQueryResponse(BaseModel):
    predicates: list[str]
    results: list[PatternMatchResponse]
