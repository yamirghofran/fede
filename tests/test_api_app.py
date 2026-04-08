"""Tests for vector_db.api.app."""

from fastapi.testclient import TestClient

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
        self._ready = ready

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
