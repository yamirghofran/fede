"""Tests for vector_db.retrieval."""

from unittest.mock import MagicMock, patch

import pytest

from vector_db.config import QdrantConfig
from vector_db.retrieval import (
    SceneResult,
    ScriptRetriever,
    SentenceResult,
    hierarchical_search,
    _hit_to_scene_result,
    _hit_to_sentence_result,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_CFG = QdrantConfig(mode="local", path="/tmp/fede_test", vector_size=4)
_QUERY = [0.1, 0.2, 0.3, 0.4]


def _mock_sentence_hit(scene_id: str, score: float, position: int = 0, movie_id: str = "m1"):
    hit = MagicMock()
    hit.id = f"sent-{scene_id}-{position}"
    hit.score = score
    hit.payload = {
        "movie_id": movie_id,
        "movie_title": "Test Movie",
        "scene_id": scene_id,
        "scene_index": 0,
        "text": "A line of dialogue.",
        "line_type": "dialogue",
        "character_name": "ALICE",
        "position_in_script": position,
    }
    return hit


def _mock_scene_hit(scene_id: str, score: float, movie_id: str = "m1"):
    hit = MagicMock()
    hit.id = f"scene-{scene_id}"
    hit.score = score
    hit.payload = {
        "movie_id": movie_id,
        "movie_title": "Test Movie",
        "scene_id": scene_id,
        "scene_index": 0,
        "text": "INT. ROOM - DAY",
        "scene_title": "INT. ROOM - DAY",
        "character_names": ["ALICE"],
    }
    return hit


def _mock_scroll_point(scene_id: str, movie_id: str = "m1"):
    point = MagicMock()
    point.id = f"scene-{scene_id}"
    point.payload = {
        "movie_id": movie_id,
        "movie_title": "Test Movie",
        "scene_id": scene_id,
        "scene_index": 0,
        "text": "INT. ROOM - DAY",
        "scene_title": "INT. ROOM - DAY",
        "character_names": [],
    }
    return point


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def retriever(mock_client):
    with patch("vector_db.retrieval.get_qdrant_client", return_value=mock_client):
        yield ScriptRetriever(_CFG)


# ---------------------------------------------------------------------------
# Payload converters
# ---------------------------------------------------------------------------

class TestHitConverters:
    def test_sentence_hit_to_result(self):
        hit = _mock_sentence_hit("sc1", 0.9, position=5)
        result = _hit_to_sentence_result(hit)
        assert isinstance(result, SentenceResult)
        assert result.score == 0.9
        assert result.scene_id == "sc1"
        assert result.position_in_script == 5
        assert result.line_type == "dialogue"
        assert result.character_name == "ALICE"

    def test_scene_hit_to_result(self):
        hit = _mock_scene_hit("sc1", 0.75)
        result = _hit_to_scene_result(hit)
        assert isinstance(result, SceneResult)
        assert result.score == 0.75
        assert result.scene_id == "sc1"
        assert result.scene_title == "INT. ROOM - DAY"
        assert "ALICE" in result.character_names

    def test_missing_payload_fields_use_defaults(self):
        hit = MagicMock()
        hit.id = "x"
        hit.score = 0.5
        hit.payload = {}
        result = _hit_to_sentence_result(hit)
        assert result.movie_id == ""
        assert result.line_type == "description"
        assert result.character_name is None

        result_scene = _hit_to_scene_result(hit)
        assert result_scene.scene_title is None
        assert result_scene.character_names == []


# ---------------------------------------------------------------------------
# search_sentences
# ---------------------------------------------------------------------------

def _mock_query_response(hits):
    """Wrap a list of hits in a mock query_points response object."""
    response = MagicMock()
    response.points = hits
    return response


class TestSearchSentences:
    def test_searches_sentences_collection(self, retriever, mock_client):
        mock_client.query_points.return_value = _mock_query_response([])
        retriever.search_sentences(_QUERY, top_k=5)
        mock_client.query_points.assert_called_once()
        assert mock_client.query_points.call_args.kwargs["collection_name"] == "sentences"

    def test_passes_top_k_as_limit(self, retriever, mock_client):
        mock_client.query_points.return_value = _mock_query_response([])
        retriever.search_sentences(_QUERY, top_k=7)
        assert mock_client.query_points.call_args.kwargs["limit"] == 7

    def test_returns_sentence_results(self, retriever, mock_client):
        mock_client.query_points.return_value = _mock_query_response([
            _mock_sentence_hit("sc1", 0.9),
            _mock_sentence_hit("sc2", 0.8),
        ])
        results = retriever.search_sentences(_QUERY, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SentenceResult) for r in results)

    def test_movie_id_filter_passed_when_provided(self, retriever, mock_client):
        mock_client.query_points.return_value = _mock_query_response([])
        retriever.search_sentences(_QUERY, top_k=5, movie_id_filter="m42")
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    def test_no_filter_when_movie_id_is_none(self, retriever, mock_client):
        mock_client.query_points.return_value = _mock_query_response([])
        retriever.search_sentences(_QUERY, top_k=5, movie_id_filter=None)
        assert mock_client.query_points.call_args.kwargs["query_filter"] is None


# ---------------------------------------------------------------------------
# search_scenes
# ---------------------------------------------------------------------------

class TestSearchScenes:
    def test_searches_scenes_collection(self, retriever, mock_client):
        mock_client.query_points.return_value = _mock_query_response([])
        retriever.search_scenes(_QUERY, top_k=5)
        assert mock_client.query_points.call_args.kwargs["collection_name"] == "scenes"

    def test_returns_scene_results(self, retriever, mock_client):
        mock_client.query_points.return_value = _mock_query_response([_mock_scene_hit("sc1", 0.9)])
        results = retriever.search_scenes(_QUERY, top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SceneResult)


# ---------------------------------------------------------------------------
# hierarchical_search — merge logic
# ---------------------------------------------------------------------------

class TestHierarchicalSearch:
    def _setup_client(self, mock_client, sentence_hits, scene_hits, scroll_points=None):
        """Wire mock_client so query_points() returns sentence then scene hits,
        and scroll() returns optional extra points for sentence-only scenes."""
        mock_client.query_points.side_effect = [
            _mock_query_response(sentence_hits),
            _mock_query_response(scene_hits),
        ]
        mock_client.scroll.return_value = (scroll_points or [], None)

    def test_returns_scene_results(self, retriever, mock_client):
        self._setup_client(
            mock_client,
            sentence_hits=[_mock_sentence_hit("sc1", 0.9)],
            scene_hits=[_mock_scene_hit("sc1", 0.8)],
        )
        results = retriever.hierarchical_search(_QUERY, top_k=5)
        assert all(isinstance(r, SceneResult) for r in results)

    def test_scene_in_both_paths_takes_max_score(self, retriever, mock_client):
        # sentence score 0.95 > scene score 0.80 → merged should be 0.95
        self._setup_client(
            mock_client,
            sentence_hits=[_mock_sentence_hit("sc1", 0.95)],
            scene_hits=[_mock_scene_hit("sc1", 0.80)],
        )
        results = retriever.hierarchical_search(_QUERY, top_k=5)
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.95)

    def test_scene_score_wins_when_higher(self, retriever, mock_client):
        # scene score 0.99 > sentence score 0.70 → merged should be 0.99
        self._setup_client(
            mock_client,
            sentence_hits=[_mock_sentence_hit("sc1", 0.70)],
            scene_hits=[_mock_scene_hit("sc1", 0.99)],
        )
        results = retriever.hierarchical_search(_QUERY, top_k=5)
        assert results[0].score == pytest.approx(0.99)

    def test_sentence_only_scene_is_included(self, retriever, mock_client):
        # sc2 reached only via sentence path; sc1 via both.
        self._setup_client(
            mock_client,
            sentence_hits=[
                _mock_sentence_hit("sc1", 0.9),
                _mock_sentence_hit("sc2", 0.85),
            ],
            scene_hits=[_mock_scene_hit("sc1", 0.8)],
            scroll_points=[_mock_scroll_point("sc2")],
        )
        results = retriever.hierarchical_search(_QUERY, top_k=5)
        scene_ids = {r.scene_id for r in results}
        assert "sc2" in scene_ids

    def test_sentence_only_scene_score_is_sentence_score(self, retriever, mock_client):
        self._setup_client(
            mock_client,
            sentence_hits=[_mock_sentence_hit("sc2", 0.85)],
            scene_hits=[],
            scroll_points=[_mock_scroll_point("sc2")],
        )
        results = retriever.hierarchical_search(_QUERY, top_k=5)
        assert results[0].score == pytest.approx(0.85)

    def test_results_sorted_by_descending_score(self, retriever, mock_client):
        self._setup_client(
            mock_client,
            sentence_hits=[
                _mock_sentence_hit("sc1", 0.6),
                _mock_sentence_hit("sc2", 0.9),
                _mock_sentence_hit("sc3", 0.75),
            ],
            scene_hits=[
                _mock_scene_hit("sc1", 0.6),
                _mock_scene_hit("sc2", 0.9),
                _mock_scene_hit("sc3", 0.75),
            ],
        )
        results = retriever.hierarchical_search(_QUERY, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, retriever, mock_client):
        self._setup_client(
            mock_client,
            sentence_hits=[_mock_sentence_hit(f"sc{i}", 0.9 - i * 0.05) for i in range(6)],
            scene_hits=[_mock_scene_hit(f"sc{i}", 0.9 - i * 0.05) for i in range(6)],
        )
        results = retriever.hierarchical_search(_QUERY, top_k=3)
        assert len(results) <= 3

    def test_multiple_sentences_same_scene_takes_max(self, retriever, mock_client):
        # Two sentences in sc1 with scores 0.6 and 0.95 — max should be 0.95
        self._setup_client(
            mock_client,
            sentence_hits=[
                _mock_sentence_hit("sc1", 0.6, position=0),
                _mock_sentence_hit("sc1", 0.95, position=1),
            ],
            scene_hits=[_mock_scene_hit("sc1", 0.5)],
        )
        results = retriever.hierarchical_search(_QUERY, top_k=5)
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.95)

    def test_scroll_called_only_for_sentence_only_scenes(self, retriever, mock_client):
        # sc1 in both paths, sc2 only in sentence path → scroll needed for sc2
        self._setup_client(
            mock_client,
            sentence_hits=[
                _mock_sentence_hit("sc1", 0.9),
                _mock_sentence_hit("sc2", 0.8),
            ],
            scene_hits=[_mock_scene_hit("sc1", 0.85)],
            scroll_points=[_mock_scroll_point("sc2")],
        )
        retriever.hierarchical_search(_QUERY, top_k=5)
        mock_client.scroll.assert_called_once()

    def test_no_scroll_when_all_scenes_in_direct_results(self, retriever, mock_client):
        self._setup_client(
            mock_client,
            sentence_hits=[_mock_sentence_hit("sc1", 0.9)],
            scene_hits=[_mock_scene_hit("sc1", 0.85)],
        )
        retriever.hierarchical_search(_QUERY, top_k=5)
        mock_client.scroll.assert_not_called()

    def test_empty_results_when_no_hits(self, retriever, mock_client):
        self._setup_client(mock_client, sentence_hits=[], scene_hits=[])
        results = retriever.hierarchical_search(_QUERY, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Module-level convenience wrapper
# ---------------------------------------------------------------------------

class TestHierarchicalSearchWrapper:
    def test_delegates_to_retriever(self, mock_client):
        mock_client.query_points.side_effect = [
            _mock_query_response([]),
            _mock_query_response([]),
        ]
        mock_client.scroll.return_value = ([], None)
        with patch("vector_db.retrieval.get_qdrant_client", return_value=mock_client):
            results = hierarchical_search(_QUERY, top_k=5, config=_CFG)
        assert isinstance(results, list)
