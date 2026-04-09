"""Hybrid natural-language query pipeline over semantic and graph search."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from knowledge_graph.graph_models import PatternMatchResponse, PatternQueryRequest, PatternQueryResponse
from knowledge_graph.predicates import VALID_ENTITY_TYPES, VALID_PREDICATES

from .models import (
    GraphTranslationResponse,
    HybridMovieResult,
    HybridQueryRequest,
    HybridQueryResponse,
    PatternMatchSummary,
    SceneSnippet,
)
from .search import MovieSearchResult, SemanticSearchService
from .settings import BackendSettings


@dataclass
class TranslationResult:
    status: str
    pattern: Optional[PatternQueryRequest] = None
    error: Optional[str] = None


class QueryTranslator:
    """Translates natural language into the graph pattern DSL."""

    def __init__(self, settings: BackendSettings):
        self._settings = settings
        self._client: Optional[OpenAI] = None

    def is_available(self) -> bool:
        return bool(self._settings.llm_api_key and self._settings.llm_model)

    def translate(self, query: str, graph_limit: int) -> TranslationResult:
        if not self.is_available():
            return TranslationResult(status="unavailable", error="LLM translation is not configured")
        try:
            raw = self._call_llm(query=query, graph_limit=graph_limit)
            payload = self._extract_json(raw)
            if payload.get("skip_graph"):
                return TranslationResult(status="skipped", error=payload.get("reason"))
            pattern = PatternQueryRequest(
                predicates=payload.get("predicates") or [],
                entity_types=payload.get("entity_types"),
                movie_ids=payload.get("movie_ids"),
                contains_entities=payload.get("contains_entities"),
                limit=payload.get("limit") or graph_limit,
            )
            return TranslationResult(status="translated", pattern=pattern)
        except Exception as exc:
            return TranslationResult(status="failed", error=str(exc))

    def _call_llm(self, *, query: str, graph_limit: int) -> str:
        if self._client is None:
            self._client = OpenAI(
                api_key=self._settings.llm_api_key,
                base_url=self._settings.llm_api_url,
            )
        predicates = ", ".join(sorted(VALID_PREDICATES))
        entity_types = ", ".join(sorted(VALID_ENTITY_TYPES))
        prompt = f"""You translate user movie-story questions into a constrained graph query schema.

Return only valid JSON. No markdown.

Allowed output fields:
- predicates: ordered list of allowed predicates
- entity_types: optional list of allowed entity types
- movie_ids: optional list of exact movie ids if the user explicitly names them
- contains_entities: optional list of exact entity names if the user explicitly names them
- limit: integer
- skip_graph: boolean
- reason: optional string

Rules:
- Use only these predicates: {predicates}
- Use only these entity types: {entity_types}
- If the user query is too vague for a graph motif, return {{"skip_graph": true, "reason": "..."}}.
- Prefer graph patterns only when the intent is narrative or relational.
- Do not invent movie_ids or entity names unless they are explicitly present in the question.
- Set limit to at most {graph_limit}.

User query:
{query}
"""
        response = self._client.chat.completions.create(
            model=self._settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def _extract_json(self, raw: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)


class HybridQueryService:
    """Broker that merges semantic retrieval with graph pattern retrieval."""

    def __init__(
        self,
        *,
        settings: BackendSettings,
        semantic_service: Optional[SemanticSearchService],
        graph_service,
    ):
        self._settings = settings
        self._semantic_service = semantic_service
        self._graph_service = graph_service
        self._translator = QueryTranslator(settings)

    def query(self, body: HybridQueryRequest) -> HybridQueryResponse:
        resolved_top_k = self._settings.default_top_k if body.top_k is None else body.top_k
        resolved_sentence_pool = (
            self._settings.default_sentence_pool
            if body.sentence_pool is None
            else body.sentence_pool
        )
        graph_limit = body.graph_limit or resolved_top_k

        semantic_hits: list[MovieSearchResult] = []
        if body.use_semantic and self._semantic_service is not None:
            semantic_hits = self._semantic_service.search_movies(
                query=body.query,
                top_k=resolved_top_k,
                sentence_pool=resolved_sentence_pool,
            )

        translation = TranslationResult(status="skipped", error="graph search disabled")
        graph_hits = PatternQueryResponse(predicates=[], results=[])
        if body.use_graph and self._graph_service is not None:
            translation = self._translator.translate(body.query, graph_limit)
            if translation.pattern is not None:
                graph_hits = self._graph_service.query_pattern(translation.pattern)

        merged = self._merge_hits(
            semantic_hits=semantic_hits,
            graph_hits=graph_hits.results,
            top_k=resolved_top_k,
        )
        strategy = self._strategy_for(
            semantic_hits=semantic_hits,
            graph_hits=graph_hits.results,
            body=body,
            results=merged,
        )
        return HybridQueryResponse(
            query=body.query,
            top_k=resolved_top_k,
            sentence_pool=resolved_sentence_pool,
            graph_limit=graph_limit,
            strategy=strategy,
            translation=GraphTranslationResponse(
                status=translation.status,
                pattern=translation.pattern,
                error=translation.error,
            ),
            results=merged,
        )

    def _merge_hits(
        self,
        *,
        semantic_hits: list[MovieSearchResult],
        graph_hits: list[PatternMatchResponse],
        top_k: int,
    ) -> list[HybridMovieResult]:
        semantic_max = semantic_hits[0].score if semantic_hits else 1.0
        graph_max = max((hit.score for hit in graph_hits), default=1.0)
        merged: dict[str, HybridMovieResult] = {}

        for hit in semantic_hits:
            merged[hit.movie_id] = HybridMovieResult(
                rank=0,
                movie_id=hit.movie_id,
                movie_title=hit.movie_title,
                score=hit.score / semantic_max if semantic_max else hit.score,
                semantic_score=hit.score,
                graph_score=None,
                best_scene=SceneSnippet(
                    point_id=hit.best_scene.point_id,
                    scene_id=hit.best_scene.scene_id,
                    scene_index=hit.best_scene.scene_index,
                    scene_title=hit.best_scene.scene_title,
                    score=hit.best_scene.score,
                    text=hit.best_scene.text,
                    character_names=hit.best_scene.character_names,
                ),
                graph_matches=[],
            )

        for match in graph_hits:
            existing = merged.get(match.movie_id)
            graph_score = match.score / graph_max if graph_max else match.score
            summary = PatternMatchSummary(
                score=match.score,
                path_entities=[step.entity_name for step in match.path],
                predicates=[item.predicate for item in match.evidences],
                evidences=[item.evidence for item in match.evidences],
            )
            if existing is None:
                merged[match.movie_id] = HybridMovieResult(
                    rank=0,
                    movie_id=match.movie_id,
                    movie_title=match.movie_title,
                    score=graph_score,
                    semantic_score=None,
                    graph_score=match.score,
                    best_scene=None,
                    graph_matches=[summary],
                )
                continue
            existing.graph_score = match.score
            existing.graph_matches.append(summary)
            existing.score = (existing.score + graph_score) / 2.0 if existing.semantic_score is not None else graph_score

        ranked = sorted(
            merged.values(),
            key=lambda item: (item.score, item.graph_score or 0.0, item.semantic_score or 0.0),
            reverse=True,
        )[:top_k]
        for rank, item in enumerate(ranked, start=1):
            item.rank = rank
        return ranked

    def _strategy_for(
        self,
        *,
        semantic_hits: list[MovieSearchResult],
        graph_hits: list[PatternMatchResponse],
        body: HybridQueryRequest,
        results: list[HybridMovieResult],
    ) -> str:
        if not results:
            return "no_results"
        if body.use_graph and body.use_semantic and semantic_hits and graph_hits:
            return "hybrid"
        if semantic_hits:
            return "semantic_only"
        return "graph_only"
