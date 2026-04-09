"""Tests for vector_db.api.app."""

from fastapi.testclient import TestClient

from knowledge_graph.graph_models import GraphBuildResponse, GraphHealthResponse, MovieGraphResponse, PatternQueryResponse
from knowledge_graph.graph_store import KnowledgeGraphService, MemoryGraphBackend
from vector_db.api.app import create_app
from vector_db.api.search import MovieSearchResult
from vector_db.api.settings import BackendSettings
from vector_db.retrieval import SceneResult


def _settings() -> BackendSettings:
    return BackendSettings(
        default_top_k=3,
        max_top_k=10,
        default_sentence_pool=50,
        max_sentence_pool=200,
    )


def _scene(scene_id: str, score: float) -> SceneResult:
    return SceneResult(
        point_id=f"point-{scene_id}",
        score=score,
        movie_id="movie-1",
        movie_title="Movie 1",
        scene_id=scene_id,
        scene_index=1,
        text=f"text for {scene_id}",
        scene_title=f"title {scene_id}",
        character_names=["ALICE", "BOB"],
    )


class FakeEmbedder:
    def info(self) -> dict:
        return {
            "loaded": True,
            "model_id": "test-model",
            "vector_size": 4,
            "device": "cpu",
        }


class FakeSearchService:
    def __init__(self):
        self.embedder = FakeEmbedder()

    def search_movies(self, query: str, top_k=None, sentence_pool=None):
        return [
            MovieSearchResult(
                movie_id="movie-1",
                movie_title="Movie 1",
                score=0.91,
                best_scene=_scene("scene-a", 0.91),
            ),
            MovieSearchResult(
                movie_id="movie-2",
                movie_title="Movie 2",
                score=0.85,
                best_scene=SceneResult(
                    point_id="point-scene-b",
                    score=0.85,
                    movie_id="movie-2",
                    movie_title="Movie 2",
                    scene_id="scene-b",
                    scene_index=2,
                    text="text for scene-b",
                    scene_title="title scene-b",
                    character_names=[],
                ),
            ),
        ][:top_k or 3]

    def search_scenes(self, query: str, top_k=None, sentence_pool=None):
        return [_scene("scene-a", 0.91), _scene("scene-b", 0.84)][:top_k or 3]


class FakeRuntime:
    def __init__(self, ready: bool):
        self.settings = _settings()
        self.search_service = FakeSearchService() if ready else None
        self.graph_service = FakeGraphService()
        self._ready = ready
        self.graph_startup_error = None

    def initialize(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def ensure_ready(self):
        if not self._ready or self.search_service is None:
            raise RuntimeError("backend not ready")
        return self.search_service

    def readiness(self) -> dict:
        return {
            "status": "ready" if self._ready else "not_ready",
            "collections": {"scenes": self._ready, "sentences": self._ready},
            "model": (
                self.search_service.embedder.info()
                if self.search_service is not None
                else {"loaded": False, "model_id": "test-model"}
            ),
            "qdrant": {"connected": self._ready, "mode": "server", "config": None},
            "error": None if self._ready else "backend not ready",
        }

    def ensure_graph_ready(self):
        return self.graph_service


class FakeGraphService:
    def health(self) -> GraphHealthResponse:
        return GraphHealthResponse(
            status="ready",
            db_path="/tmp/story_graph.db",
            grafeo_available=True,
            counts={"movies": 1, "entities": 3, "narrative_edges": 2, "total_edges": 5},
            last_build=None,
            error=None,
        )

    def build(self, movie_id=None, rebuild=False) -> GraphBuildResponse:
        return GraphBuildResponse(
            mode="movie_reload" if movie_id else "full_rebuild",
            requested_movie_id=movie_id,
            movies_loaded=1,
            nodes_created=4,
            edges_created=5,
            dropped_relations=0,
            dropped_by_reason={},
            db_path="/tmp/story_graph.db",
        )

    def movie_details(self, movie_id: str) -> MovieGraphResponse:
        return MovieGraphResponse(
            movie_id=movie_id,
            title="Movie 1",
            source_file="movie-1.txt",
            entities=[
                {"entity_id": "movie-1:person:alice", "canonical_name": "Alice", "entity_type": "PERSON"},
                {"entity_id": "movie-1:person:bob", "canonical_name": "Bob", "entity_type": "PERSON"},
            ],
            outgoing_relations=[
                {
                    "relation_id": "rel-1",
                    "predicate": "TEACHES",
                    "from_entity_id": "movie-1:person:alice",
                    "from_name": "Alice",
                    "to_entity_id": "movie-1:person:bob",
                    "to_name": "Bob",
                    "evidence": "Alice teaches Bob.",
                }
            ],
            incoming_relations=[],
        )

    def query_pattern(self, _body) -> PatternQueryResponse:
        return PatternQueryResponse(
            predicates=["TEACHES", "BETRAYS"],
            results=[
                {
                    "movie_id": "movie-1",
                    "movie_title": "Movie 1",
                    "score": 2.03,
                    "path": [
                        {"entity_id": "a", "entity_name": "Alice", "entity_type": "PERSON"},
                        {"entity_id": "b", "entity_name": "Bob", "entity_type": "PERSON"},
                        {"entity_id": "c", "entity_name": "Carol", "entity_type": "PERSON"},
                    ],
                    "evidences": [
                        {"relation_id": "rel-1", "predicate": "TEACHES", "evidence": "Alice teaches Bob."},
                        {"relation_id": "rel-2", "predicate": "BETRAYS", "evidence": "Bob betrays Carol."},
                    ],
                }
            ],
        )


def test_healthz_returns_ok():
    app = create_app(settings=_settings(), runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_ready():
    app = create_app(settings=_settings(), runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_search_returns_ranked_movies():
    app = create_app(settings=_settings(), runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.post("/search", json={"query": "missing child", "top_k": 2})

    payload = response.json()
    assert response.status_code == 200
    assert payload["top_k"] == 2
    assert [hit["rank"] for hit in payload["results"]] == [1, 2]
    assert payload["results"][0]["movie_id"] == "movie-1"
    assert payload["results"][0]["best_scene"]["scene_id"] == "scene-a"


def test_search_scenes_returns_ranked_scenes():
    app = create_app(settings=_settings(), runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.post("/search/scenes", json={"query": "missing child", "top_k": 2})

    payload = response.json()
    assert response.status_code == 200
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][1]["scene_id"] == "scene-b"


def test_blank_query_returns_422():
    app = create_app(settings=_settings(), runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.post("/search", json={"query": "   "})

    assert response.status_code == 422


def test_not_ready_returns_503():
    app = create_app(settings=_settings(), runtime=FakeRuntime(ready=False))

    with TestClient(app) as client:
        ready_response = client.get("/readyz")
        search_response = client.post("/search", json={"query": "missing child"})

    assert ready_response.status_code == 503
    assert search_response.status_code == 503


def test_graph_health_returns_status():
    app = create_app(settings=_settings(), runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.get("/graph/health")

    assert response.status_code == 200
    assert response.json()["counts"]["movies"] == 1


def test_graph_build_returns_summary():
    app = create_app(settings=_settings(), runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.post("/graph/build", json={"movie_id": "movie-1"})

    assert response.status_code == 200
    assert response.json()["requested_movie_id"] == "movie-1"


def test_graph_movie_returns_entities_and_relations():
    app = create_app(settings=_settings(), runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.get("/graph/movies/movie-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["movie_id"] == "movie-1"
    assert payload["entities"][0]["canonical_name"] == "Alice"
    assert payload["outgoing_relations"][0]["predicate"] == "TEACHES"


def test_graph_pattern_query_returns_matches():
    app = create_app(settings=_settings(), runtime=FakeRuntime(ready=True))

    with TestClient(app) as client:
        response = client.post("/graph/query/pattern", json={"predicates": ["TEACHES", "BETRAYS"], "limit": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicates"] == ["TEACHES", "BETRAYS"]
    assert payload["results"][0]["path"][1]["entity_name"] == "Bob"
