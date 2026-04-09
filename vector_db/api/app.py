"""FastAPI application for FEDE semantic retrieval."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from knowledge_graph.graph_models import PatternQueryRequest
from .models import (
    GraphBuildRequest,
    GraphBuildResponse,
    GraphHealthResponse,
    HealthResponse,
    HybridQueryRequest,
    HybridQueryResponse,
    MovieGraphResponse,
    MovieSearchResponse,
    PatternQueryResponse,
    RankedMovieResult,
    RankedSceneResult,
    ReadyResponse,
    SceneSearchResponse,
    SceneSnippet,
    SearchRequest,
)
from .runtime import BackendRuntime
from .search import MovieSearchResult, SemanticSearchService
from .settings import BackendSettings


def create_app(
    settings: BackendSettings | None = None,
    runtime: BackendRuntime | None = None,
) -> FastAPI:
    resolved_settings = settings or (runtime.settings if runtime is not None else BackendSettings())
    resolved_runtime = runtime or BackendRuntime(settings=resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = resolved_runtime
        resolved_runtime.initialize()
        yield
        resolved_runtime.shutdown()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse)
    def readyz(request: Request) -> JSONResponse:
        payload = _runtime(request).readiness()
        response = ReadyResponse(**payload)
        status_code = (
            status.HTTP_200_OK
            if response.status == "ready"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return JSONResponse(status_code=status_code, content=response.model_dump())

    @app.post("/search", response_model=MovieSearchResponse)
    def search_movies(body: SearchRequest, request: Request) -> MovieSearchResponse:
        service = _service(request)
        try:
            results = service.search_movies(
                query=body.query,
                top_k=body.top_k,
                sentence_pool=body.sentence_pool,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return MovieSearchResponse(
            query=body.query,
            top_k=_resolved_top_k(body, _runtime(request).settings),
            sentence_pool=_resolved_sentence_pool(body, _runtime(request).settings),
            results=[
                _movie_result(hit, rank)
                for rank, hit in enumerate(results, start=1)
            ],
        )

    @app.post("/search/scenes", response_model=SceneSearchResponse)
    def search_scenes(body: SearchRequest, request: Request) -> SceneSearchResponse:
        service = _service(request)
        try:
            results = service.search_scenes(
                query=body.query,
                top_k=body.top_k,
                sentence_pool=body.sentence_pool,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return SceneSearchResponse(
            query=body.query,
            top_k=_resolved_top_k(body, _runtime(request).settings),
            sentence_pool=_resolved_sentence_pool(body, _runtime(request).settings),
            results=[
                _scene_result(hit, rank)
                for rank, hit in enumerate(results, start=1)
            ],
        )

    @app.post("/query", response_model=HybridQueryResponse)
    def hybrid_query(body: HybridQueryRequest, request: Request) -> HybridQueryResponse:
        service = _runtime(request).get_hybrid_service()
        try:
            return service.query(body)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.get("/graph/health", response_model=GraphHealthResponse)
    def graph_health(request: Request) -> GraphHealthResponse:
        runtime = _runtime(request)
        service = runtime.graph_service
        if service is None:
            error = None if runtime.graph_startup_error is None else str(runtime.graph_startup_error)
            return GraphHealthResponse(
                status="not_ready",
                db_path=str(runtime.settings.graph_db_path),
                grafeo_available=False,
                counts={"movies": 0, "entities": 0, "narrative_edges": 0, "total_edges": 0},
                last_build=None,
                error=error,
            )
        return service.health()

    @app.post("/graph/build", response_model=GraphBuildResponse)
    def build_graph(body: GraphBuildRequest, request: Request) -> GraphBuildResponse:
        service = _graph_service(request)
        try:
            return service.build(movie_id=body.movie_id, rebuild=body.rebuild)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.get("/graph/movies/{movie_id}", response_model=MovieGraphResponse)
    def graph_movie(movie_id: str, request: Request) -> MovieGraphResponse:
        service = _graph_service(request)
        try:
            return service.movie_details(movie_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown movie_id: {movie_id}") from exc

    @app.post("/graph/query/pattern", response_model=PatternQueryResponse)
    def graph_query_pattern(body: PatternQueryRequest, request: Request) -> PatternQueryResponse:
        service = _graph_service(request)
        return service.query_pattern(body)

    return app


def _runtime(request: Request) -> BackendRuntime:
    return request.app.state.runtime


def _service(request: Request) -> SemanticSearchService:
    runtime = _runtime(request)
    try:
        return runtime.ensure_ready()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def _graph_service(request: Request):
    runtime = _runtime(request)
    try:
        return runtime.ensure_graph_ready()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def _resolved_top_k(body: SearchRequest, settings: BackendSettings) -> int:
    return settings.default_top_k if body.top_k is None else body.top_k


def _resolved_sentence_pool(body: SearchRequest, settings: BackendSettings) -> int:
    return settings.default_sentence_pool if body.sentence_pool is None else body.sentence_pool


def _scene_snippet(point_id: str, scene_id: str, scene_index: int, scene_title: str | None, score: float, text: str, character_names: list[str]) -> SceneSnippet:
    return SceneSnippet(
        point_id=point_id,
        scene_id=scene_id,
        scene_index=scene_index,
        scene_title=scene_title,
        score=score,
        text=text,
        character_names=character_names,
    )


def _movie_result(hit: MovieSearchResult, rank: int) -> RankedMovieResult:
    best_scene = hit.best_scene
    return RankedMovieResult(
        rank=rank,
        movie_id=hit.movie_id,
        movie_title=hit.movie_title,
        score=hit.score,
        best_scene=_scene_snippet(
            point_id=best_scene.point_id,
            scene_id=best_scene.scene_id,
            scene_index=best_scene.scene_index,
            scene_title=best_scene.scene_title,
            score=best_scene.score,
            text=best_scene.text,
            character_names=best_scene.character_names,
        ),
    )


def _scene_result(hit, rank: int) -> RankedSceneResult:
    return RankedSceneResult(
        rank=rank,
        point_id=hit.point_id,
        score=hit.score,
        movie_id=hit.movie_id,
        movie_title=hit.movie_title,
        scene_id=hit.scene_id,
        scene_index=hit.scene_index,
        text=hit.text,
        scene_title=hit.scene_title,
        character_names=hit.character_names,
    )


app = create_app()
