"""Tests for vector_db.api.search."""

from unittest.mock import MagicMock

import pytest

from knowledge_graph.graph_models import PatternMatchResponse, PatternQueryRequest, PatternStep, PatternEdgeEvidence
from vector_db.api.hybrid import HybridQueryService, TranslationResult
from vector_db.api.models import HybridQueryRequest
from vector_db.api.search import SemanticSearchService
from vector_db.api.settings import BackendSettings
from vector_db.retrieval import SceneResult


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def encode_query(self, query: str):
        self.queries.append(query)
        return [0.1, 0.2, 0.3, 0.4]


def _settings() -> BackendSettings:
    return BackendSettings(
        default_top_k=2,
        max_top_k=5,
        default_sentence_pool=25,
        max_sentence_pool=100,
        movie_overfetch_factor=3,
    )


def _scene(
    movie_id: str,
    movie_title: str,
    scene_id: str,
    score: float,
) -> SceneResult:
    return SceneResult(
        point_id=f"point-{scene_id}",
        score=score,
        movie_id=movie_id,
        movie_title=movie_title,
        scene_id=scene_id,
        scene_index=0,
        text=f"text for {scene_id}",
        scene_title=f"title {scene_id}",
        character_names=["ALICE"],
    )


def test_search_movies_deduplicates_by_movie():
    retriever = MagicMock()
    retriever.hierarchical_search.return_value = [
        _scene("movie-1", "Movie 1", "scene-a", 0.91),
        _scene("movie-2", "Movie 2", "scene-b", 0.87),
        _scene("movie-1", "Movie 1", "scene-c", 0.72),
    ]
    embedder = FakeEmbedder()
    service = SemanticSearchService(retriever, embedder, _settings())

    results = service.search_movies("space opera", top_k=2, sentence_pool=40)

    assert [hit.movie_id for hit in results] == ["movie-1", "movie-2"]
    assert results[0].best_scene.scene_id == "scene-a"
    assert embedder.queries == ["space opera"]
    retriever.hierarchical_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        top_k=6,
        sentence_pool=40,
    )


def test_search_scenes_uses_defaults():
    retriever = MagicMock()
    retriever.hierarchical_search.return_value = [_scene("movie-1", "Movie 1", "scene-a", 0.91)]
    service = SemanticSearchService(retriever, FakeEmbedder(), _settings())

    results = service.search_scenes("robot detective")

    assert len(results) == 1
    retriever.hierarchical_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        top_k=2,
        sentence_pool=25,
    )


def test_search_movies_rejects_top_k_above_limit():
    service = SemanticSearchService(MagicMock(), FakeEmbedder(), _settings())

    with pytest.raises(ValueError, match="top_k"):
        service.search_movies("query", top_k=6)


def test_search_movies_rejects_sentence_pool_above_limit():
    service = SemanticSearchService(MagicMock(), FakeEmbedder(), _settings())

    with pytest.raises(ValueError, match="sentence_pool"):
        service.search_movies("query", sentence_pool=101)


class FakeGraphService:
    def query_pattern(self, request: PatternQueryRequest):
        assert request.predicates == ["TEACHES", "BETRAYS"]
        return type(
            "Response",
            (),
            {
                "results": [
                    PatternMatchResponse(
                        movie_id="movie-1",
                        movie_title="Movie 1",
                        score=2.0,
                        path=[
                            PatternStep(entity_id="a", entity_name="Alice", entity_type="PERSON"),
                            PatternStep(entity_id="b", entity_name="Bob", entity_type="PERSON"),
                            PatternStep(entity_id="c", entity_name="Carol", entity_type="PERSON"),
                        ],
                        evidences=[
                            PatternEdgeEvidence(relation_id="r1", predicate="TEACHES", evidence="Alice teaches Bob."),
                            PatternEdgeEvidence(relation_id="r2", predicate="BETRAYS", evidence="Bob betrays Carol."),
                        ],
                    )
                ]
            },
        )()


class FakeTranslator:
    def translate(self, query: str, graph_limit: int) -> TranslationResult:
        assert query == "mentor betrays student"
        assert graph_limit == 3
        return TranslationResult(
            status="translated",
            pattern=PatternQueryRequest(predicates=["TEACHES", "BETRAYS"], limit=3),
        )


def test_hybrid_query_merges_semantic_and_graph_results():
    retriever = MagicMock()
    retriever.hierarchical_search.return_value = [
        _scene("movie-1", "Movie 1", "scene-a", 0.91),
        _scene("movie-2", "Movie 2", "scene-b", 0.82),
    ]
    semantic = SemanticSearchService(retriever, FakeEmbedder(), _settings())
    service = HybridQueryService(
        settings=_settings(),
        semantic_service=semantic,
        graph_service=FakeGraphService(),
    )
    service._translator = FakeTranslator()

    response = service.query(
        HybridQueryRequest(
            query="mentor betrays student",
            top_k=3,
            graph_limit=3,
        )
    )

    assert response.strategy == "hybrid"
    assert response.translation.status == "translated"
    assert response.results[0].movie_id == "movie-1"
    assert response.results[0].graph_matches[0].predicates == ["TEACHES", "BETRAYS"]
    assert response.results[0].best_scene is not None


def test_hybrid_query_falls_back_to_semantic_only():
    retriever = MagicMock()
    retriever.hierarchical_search.return_value = [_scene("movie-1", "Movie 1", "scene-a", 0.91)]
    semantic = SemanticSearchService(retriever, FakeEmbedder(), _settings())
    service = HybridQueryService(
        settings=_settings(),
        semantic_service=semantic,
        graph_service=None,
    )

    response = service.query(HybridQueryRequest(query="find movies like whiplash", top_k=2))

    assert response.strategy == "semantic_only"
    assert response.translation.status == "skipped"
    assert response.results[0].movie_id == "movie-1"
