"""Graph storage backends and service wrapper."""

from __future__ import annotations

import gc
import importlib
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional

from .graph_models import (
    BuildSummary,
    GraphBuildResponse,
    GraphCounts,
    GraphHealthResponse,
    MovieEntityResponse,
    MovieGraphDocument,
    MovieGraphResponse,
    MovieRelationResponse,
    PatternEdgeEvidence,
    PatternMatchResponse,
    PatternQueryRequest,
    PatternQueryResponse,
    PatternStep,
)
from .graph_normalize import available_movie_ids, load_movie_document


class BaseGraphBackend(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def replace_movie(self, document: MovieGraphDocument) -> None:
        raise NotImplementedError

    @abstractmethod
    def counts(self) -> GraphCounts:
        raise NotImplementedError

    @abstractmethod
    def movie_details(self, movie_id: str, relation_limit: int = 25) -> MovieGraphResponse:
        raise NotImplementedError

    @abstractmethod
    def pattern_query(self, request: PatternQueryRequest) -> PatternQueryResponse:
        raise NotImplementedError


class GrafeoGraphBackend(BaseGraphBackend):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        grafeo = importlib.import_module("grafeo")
        self._db = grafeo.GrafeoDB(str(db_path))

    def reset(self) -> None:
        self._db = None
        gc.collect()
        if self.db_path.exists():
            if self.db_path.is_dir():
                shutil.rmtree(self.db_path)
            else:
                self.db_path.unlink()
        grafeo = importlib.import_module("grafeo")
        self._db = grafeo.GrafeoDB(str(self.db_path))

    def replace_movie(self, document: MovieGraphDocument) -> None:
        with self._db.begin_transaction() as tx:
            tx.execute_cypher("MATCH (n {movie_id: $movie_id})-[r]-() DELETE r", {"movie_id": document.movie_id})
            tx.execute_cypher("MATCH (n {movie_id: $movie_id}) DELETE n", {"movie_id": document.movie_id})
            tx.execute_cypher(
                """
                CREATE (:Movie {
                    movie_id: $movie_id,
                    title: $title,
                    source_file: $source_file
                })
                """,
                {
                    "movie_id": document.movie_id,
                    "title": document.title,
                    "source_file": document.source_file,
                },
            )
            for entity in document.entities:
                tx.execute_cypher(
                    """
                    CREATE (:Entity {
                        entity_id: $entity_id,
                        movie_id: $movie_id,
                        canonical_name: $canonical_name,
                        entity_type: $entity_type,
                        source_file: $source_file
                    })
                    """,
                    {
                        "entity_id": entity.entity_id,
                        "movie_id": entity.movie_id,
                        "canonical_name": entity.canonical_name,
                        "entity_type": entity.entity_type,
                        "source_file": entity.source_file,
                    },
                )
            for entity in document.entities:
                tx.execute_cypher(
                    """
                    MATCH (m:Movie {movie_id: $movie_id}), (e:Entity {entity_id: $entity_id})
                    CREATE (m)-[:HAS_ENTITY {movie_id: $movie_id}]->(e)
                    """,
                    {"movie_id": document.movie_id, "entity_id": entity.entity_id},
                )
            for relation in document.relations:
                tx.execute_cypher(
                    f"""
                    MATCH (a:Entity {{entity_id: $from_entity_id}}), (b:Entity {{entity_id: $to_entity_id}})
                    CREATE (a)-[:{relation.predicate} {{
                        relation_id: $relation_id,
                        movie_id: $movie_id,
                        evidence: $evidence,
                        source_file: $source_file
                    }}]->(b)
                    """,
                    {
                        "from_entity_id": relation.from_entity_id,
                        "to_entity_id": relation.to_entity_id,
                        "relation_id": relation.relation_id,
                        "movie_id": relation.movie_id,
                        "evidence": relation.evidence,
                        "source_file": relation.source_file,
                    },
                )
            tx.commit()

    def counts(self) -> GraphCounts:
        return GraphCounts(
            movies=self._single_int("MATCH (m:Movie) RETURN count(m) AS count"),
            entities=self._single_int("MATCH (e:Entity) RETURN count(e) AS count"),
            narrative_edges=self._single_int("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS count"),
            total_edges=self._single_int("MATCH ()-[r]->() RETURN count(r) AS count"),
        )

    def movie_details(self, movie_id: str, relation_limit: int = 25) -> MovieGraphResponse:
        movie_rows = list(
            self._db.execute_cypher(
                "MATCH (m:Movie {movie_id: $movie_id}) RETURN m.movie_id AS movie_id, m.title AS title, m.source_file AS source_file",
                {"movie_id": movie_id},
            )
        )
        if not movie_rows:
            raise KeyError(movie_id)
        movie = movie_rows[0]
        entities = [
            MovieEntityResponse(
                entity_id=row["entity_id"],
                canonical_name=row["canonical_name"],
                entity_type=row["entity_type"],
            )
            for row in self._db.execute_cypher(
                """
                MATCH (m:Movie {movie_id: $movie_id})-[:HAS_ENTITY]->(e:Entity)
                RETURN e.entity_id AS entity_id, e.canonical_name AS canonical_name, e.entity_type AS entity_type
                ORDER BY e.canonical_name
                """,
                {"movie_id": movie_id},
            )
        ]
        outgoing = self._relation_rows(
            """
            MATCH (a:Entity {movie_id: $movie_id})-[r]->(b:Entity {movie_id: $movie_id})
            RETURN r.relation_id AS relation_id, type(r) AS predicate,
                   a.entity_id AS from_entity_id, a.canonical_name AS from_name,
                   b.entity_id AS to_entity_id, b.canonical_name AS to_name,
                   r.evidence AS evidence
            ORDER BY a.canonical_name, predicate, b.canonical_name
            LIMIT $limit
            """,
            movie_id,
            relation_limit,
        )
        incoming = self._relation_rows(
            """
            MATCH (a:Entity {movie_id: $movie_id})-[r]->(b:Entity {movie_id: $movie_id})
            RETURN r.relation_id AS relation_id, type(r) AS predicate,
                   a.entity_id AS from_entity_id, a.canonical_name AS from_name,
                   b.entity_id AS to_entity_id, b.canonical_name AS to_name,
                   r.evidence AS evidence
            ORDER BY b.canonical_name, predicate, a.canonical_name
            LIMIT $limit
            """,
            movie_id,
            relation_limit,
        )
        return MovieGraphResponse(
            movie_id=movie["movie_id"],
            title=movie["title"],
            source_file=movie["source_file"],
            entities=entities,
            outgoing_relations=outgoing,
            incoming_relations=incoming,
        )

    def pattern_query(self, request: PatternQueryRequest) -> PatternQueryResponse:
        alias_parts = []
        return_parts = ["m.movie_id AS movie_id", "m.title AS movie_title"]
        where_parts = ["n0.movie_id = m.movie_id"]
        params: dict[str, Any] = {"limit": request.limit}
        for index, predicate in enumerate(request.predicates):
            next_index = index + 1
            alias_parts.append(f"(n{index}:Entity)-[r{index}:{predicate}]->(n{next_index}:Entity)")
            where_parts.append(f"n{next_index}.movie_id = m.movie_id")
            return_parts.extend(
                [
                    f"n{index}.entity_id AS entity_id_{index}",
                    f"n{index}.canonical_name AS entity_name_{index}",
                    f"n{index}.entity_type AS entity_type_{index}",
                    f"r{index}.relation_id AS relation_id_{index}",
                    f"type(r{index}) AS predicate_{index}",
                    f"r{index}.evidence AS evidence_{index}",
                ]
            )
        last_index = len(request.predicates)
        return_parts.extend(
            [
                f"n{last_index}.entity_id AS entity_id_{last_index}",
                f"n{last_index}.canonical_name AS entity_name_{last_index}",
                f"n{last_index}.entity_type AS entity_type_{last_index}",
            ]
        )
        if request.movie_ids:
            where_parts.append("m.movie_id IN $movie_ids")
            params["movie_ids"] = request.movie_ids
        if request.entity_types:
            for index in range(last_index + 1):
                where_parts.append(f"n{index}.entity_type IN $entity_types")
            params["entity_types"] = request.entity_types
        if request.contains_entities:
            contain_checks = []
            params["contains_entities"] = request.contains_entities
            for index in range(last_index + 1):
                contain_checks.append(f"n{index}.canonical_name IN $contains_entities")
            where_parts.append("(" + " OR ".join(contain_checks) + ")")
        query = f"""
        MATCH (m:Movie), {", ".join(alias_parts)}
        WHERE {" AND ".join(where_parts)}
        RETURN {", ".join(return_parts)}
        LIMIT $limit
        """
        rows = list(self._db.execute_cypher(query, params))
        results: list[PatternMatchResponse] = []
        for row in rows:
            path = [
                PatternStep(
                    entity_id=row[f"entity_id_{index}"],
                    entity_name=row[f"entity_name_{index}"],
                    entity_type=row[f"entity_type_{index}"],
                )
                for index in range(last_index + 1)
            ]
            evidences = [
                PatternEdgeEvidence(
                    relation_id=row[f"relation_id_{index}"],
                    predicate=row[f"predicate_{index}"],
                    evidence=row[f"evidence_{index}"],
                )
                for index in range(len(request.predicates))
            ]
            score = float(len(request.predicates)) + (len({step.entity_id for step in path}) / 100.0)
            results.append(
                PatternMatchResponse(
                    movie_id=row["movie_id"],
                    movie_title=row["movie_title"],
                    score=score,
                    path=path,
                    evidences=evidences,
                )
            )
        return PatternQueryResponse(predicates=request.predicates, results=results)

    def _relation_rows(self, query: str, movie_id: str, limit: int) -> list[MovieRelationResponse]:
        return [
            MovieRelationResponse(**row)
            for row in self._db.execute_cypher(query, {"movie_id": movie_id, "limit": limit})
        ]

    def _single_int(self, query: str) -> int:
        rows = list(self._db.execute_cypher(query))
        if not rows:
            return 0
        value = rows[0].get("count", 0)
        return int(value)


class MemoryGraphBackend(BaseGraphBackend):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.movies: dict[str, MovieGraphDocument] = {}

    def replace_movie(self, document: MovieGraphDocument) -> None:
        self.movies[document.movie_id] = document

    def counts(self) -> GraphCounts:
        entities = sum(len(document.entities) for document in self.movies.values())
        narrative_edges = sum(len(document.relations) for document in self.movies.values())
        total_edges = entities + narrative_edges
        return GraphCounts(movies=len(self.movies), entities=entities, narrative_edges=narrative_edges, total_edges=total_edges)

    def movie_details(self, movie_id: str, relation_limit: int = 25) -> MovieGraphResponse:
        if movie_id not in self.movies:
            raise KeyError(movie_id)
        document = self.movies[movie_id]
        relations = [
            MovieRelationResponse(
                relation_id=relation.relation_id,
                predicate=relation.predicate,
                from_entity_id=relation.from_entity_id,
                from_name=relation.from_name,
                to_entity_id=relation.to_entity_id,
                to_name=relation.to_name,
                evidence=relation.evidence,
            )
            for relation in document.relations[:relation_limit]
        ]
        incoming = sorted(relations, key=lambda relation: (relation.to_name, relation.predicate, relation.from_name))[:relation_limit]
        outgoing = sorted(relations, key=lambda relation: (relation.from_name, relation.predicate, relation.to_name))[:relation_limit]
        return MovieGraphResponse(
            movie_id=document.movie_id,
            title=document.title,
            source_file=document.source_file,
            entities=[
                MovieEntityResponse(
                    entity_id=entity.entity_id,
                    canonical_name=entity.canonical_name,
                    entity_type=entity.entity_type,
                )
                for entity in document.entities
            ],
            outgoing_relations=outgoing,
            incoming_relations=incoming,
        )

    def pattern_query(self, request: PatternQueryRequest) -> PatternQueryResponse:
        results: list[PatternMatchResponse] = []
        for document in self.movies.values():
            if request.movie_ids and document.movie_id not in request.movie_ids:
                continue
            entity_by_id = {entity.entity_id: entity for entity in document.entities}
            adjacency: dict[str, list[Any]] = {}
            for relation in document.relations:
                adjacency.setdefault(relation.from_entity_id, []).append(relation)
            for entity in document.entities:
                self._walk_pattern(
                    document=document,
                    request=request,
                    entity=entity,
                    adjacency=adjacency,
                    entity_by_id=entity_by_id,
                    path=[entity],
                    evidences=[],
                    depth=0,
                    results=results,
                )
                if len(results) >= request.limit:
                    break
            if len(results) >= request.limit:
                break
        return PatternQueryResponse(predicates=request.predicates, results=results[: request.limit])

    def _walk_pattern(
        self,
        *,
        document: MovieGraphDocument,
        request: PatternQueryRequest,
        entity,
        adjacency: dict[str, list[Any]],
        entity_by_id: dict[str, Any],
        path: list[Any],
        evidences: list[PatternEdgeEvidence],
        depth: int,
        results: list[PatternMatchResponse],
    ) -> None:
        if depth == len(request.predicates):
            if request.contains_entities:
                names = {item.canonical_name for item in path}
                if not any(name in names for name in request.contains_entities):
                    return
            if request.entity_types:
                if any(item.entity_type not in request.entity_types for item in path):
                    return
            score = float(len(request.predicates)) + (len({item.entity_id for item in path}) / 100.0)
            results.append(
                PatternMatchResponse(
                    movie_id=document.movie_id,
                    movie_title=document.title,
                    score=score,
                    path=[
                        PatternStep(
                            entity_id=item.entity_id,
                            entity_name=item.canonical_name,
                            entity_type=item.entity_type,
                        )
                        for item in path
                    ],
                    evidences=evidences.copy(),
                )
            )
            return
        predicate = request.predicates[depth]
        for relation in adjacency.get(entity.entity_id, []):
            if relation.predicate != predicate:
                continue
            next_entity = entity_by_id[relation.to_entity_id]
            path.append(next_entity)
            evidences.append(
                PatternEdgeEvidence(
                    relation_id=relation.relation_id,
                    predicate=relation.predicate,
                    evidence=relation.evidence,
                )
            )
            self._walk_pattern(
                document=document,
                request=request,
                entity=next_entity,
                adjacency=adjacency,
                entity_by_id=entity_by_id,
                path=path,
                evidences=evidences,
                depth=depth + 1,
                results=results,
            )
            evidences.pop()
            path.pop()


class KnowledgeGraphService:
    def __init__(
        self,
        *,
        db_path: Path,
        entities_dir: Path,
        relations_dir: Path,
        backend: Optional[BaseGraphBackend] = None,
    ):
        self.db_path = db_path
        self.entities_dir = entities_dir
        self.relations_dir = relations_dir
        self.meta_path = db_path.parent / "build_stats.json"
        self._backend = backend
        self._backend_error: Optional[str] = None
        self._last_build: Optional[BuildSummary] = self._read_last_build()

    def initialize(self) -> None:
        if self._backend is not None:
            return
        try:
            self._backend = GrafeoGraphBackend(self.db_path)
            self._backend_error = None
        except Exception as exc:
            self._backend = None
            self._backend_error = str(exc)

    @property
    def is_ready(self) -> bool:
        self.initialize()
        return self._backend is not None

    def build(self, movie_id: str | None = None, rebuild: bool = False) -> GraphBuildResponse:
        backend = self._ensure_backend()
        if movie_id is not None and rebuild:
            raise ValueError("rebuild=true is only valid for a full graph rebuild")
        movie_ids = [movie_id] if movie_id else available_movie_ids(self.entities_dir)
        if not movie_ids:
            raise ValueError("No knowledge graph entity files found")
        documents = [
            load_movie_document(item, self.entities_dir, self.relations_dir)
            for item in movie_ids
        ]
        if movie_id is None and rebuild:
            backend.reset()
        for document in documents:
            backend.replace_movie(document)
        summary = BuildSummary.from_documents(
            mode="movie_reload" if movie_id else "full_rebuild",
            requested_movie_id=movie_id,
            documents=documents,
            db_path=self.db_path,
        )
        self._last_build = summary
        self._write_last_build(summary)
        return GraphBuildResponse(**summary.model_dump())

    def health(self) -> GraphHealthResponse:
        self.initialize()
        counts = self._backend.counts() if self._backend is not None else GraphCounts()
        return GraphHealthResponse(
            status="ready" if self._backend is not None else "not_ready",
            db_path=str(self.db_path),
            grafeo_available=self._backend is not None,
            counts=counts,
            last_build=self._last_build,
            error=self._backend_error,
        )

    def movie_details(self, movie_id: str, relation_limit: int = 25) -> MovieGraphResponse:
        return self._ensure_backend().movie_details(movie_id=movie_id, relation_limit=relation_limit)

    def query_pattern(self, request: PatternQueryRequest) -> PatternQueryResponse:
        return self._ensure_backend().pattern_query(request)

    def _ensure_backend(self) -> BaseGraphBackend:
        self.initialize()
        if self._backend is None:
            raise RuntimeError(self._backend_error or "Grafeo backend is not available")
        return self._backend

    def _read_last_build(self) -> Optional[BuildSummary]:
        if not self.meta_path.exists():
            return None
        try:
            payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return BuildSummary.model_validate(payload)
        except Exception:
            return None

    def _write_last_build(self, summary: BuildSummary) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(summary.model_dump(), indent=2), encoding="utf-8")
