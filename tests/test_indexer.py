"""Tests for vector_db.indexer."""

import uuid
from unittest.mock import MagicMock, call, patch

import pytest

from vector_db.indexer import (
    SceneRecord,
    ScriptIndexer,
    SentenceRecord,
    _scene_point_id,
    _sentence_point_id,
    index_movie,
)
from vector_db.config import QdrantConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CFG = QdrantConfig(mode="local", path="/tmp/fede_test", vector_size=4)
_EMB = [0.1, 0.2, 0.3, 0.4]  # 4-dimensional stub embedding


def _make_scene(movie_id="m1", scene_id="s1", scene_index=0) -> SceneRecord:
    return SceneRecord(
        movie_id=movie_id,
        movie_title="Test Movie",
        scene_id=scene_id,
        scene_index=scene_index,
        text="INT. OFFICE - DAY\nSomeone sits.",
        embedding=_EMB,
        scene_title="INT. OFFICE - DAY",
        character_names=["ALICE"],
    )


def _make_sentence(
    movie_id="m1", scene_id="s1", scene_index=0, position=0
) -> SentenceRecord:
    return SentenceRecord(
        movie_id=movie_id,
        movie_title="Test Movie",
        scene_id=scene_id,
        scene_index=scene_index,
        text="Hello world.",
        line_type="dialogue",
        position_in_script=position,
        embedding=_EMB,
        character_name="ALICE",
    )


@pytest.fixture
def mock_client():
    """A MagicMock standing in for QdrantClient."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    return client


@pytest.fixture
def indexer(mock_client):
    with patch("vector_db.indexer.get_qdrant_client", return_value=mock_client):
        yield ScriptIndexer(_CFG)


# ---------------------------------------------------------------------------
# Deterministic ID helpers
# ---------------------------------------------------------------------------

class TestPointIdHelpers:
    def test_scene_id_is_valid_uuid(self):
        pid = _scene_point_id("movie1", "scene1")
        uuid.UUID(pid)  # raises if invalid

    def test_scene_id_is_stable(self):
        assert _scene_point_id("movie1", "scene1") == _scene_point_id("movie1", "scene1")

    def test_scene_id_differs_for_different_inputs(self):
        assert _scene_point_id("movie1", "scene1") != _scene_point_id("movie1", "scene2")
        assert _scene_point_id("movie1", "scene1") != _scene_point_id("movie2", "scene1")

    def test_sentence_id_is_valid_uuid(self):
        pid = _sentence_point_id("movie1", "scene1", 5)
        uuid.UUID(pid)

    def test_sentence_id_is_stable(self):
        assert (
            _sentence_point_id("movie1", "scene1", 5)
            == _sentence_point_id("movie1", "scene1", 5)
        )

    def test_sentence_id_differs_by_position(self):
        assert (
            _sentence_point_id("movie1", "scene1", 5)
            != _sentence_point_id("movie1", "scene1", 6)
        )

    def test_scene_and_sentence_ids_never_collide(self):
        # Even with the same movie_id and scene_id the two namespaces produce
        # different IDs because the sentence key includes a position component.
        scene_pid = _scene_point_id("m1", "s1")
        # No position can make sentence match scene because scene key has no "::<int>" suffix.
        sentence_pid = _sentence_point_id("m1", "s1", 0)
        assert scene_pid != sentence_pid


# ---------------------------------------------------------------------------
# index_scene
# ---------------------------------------------------------------------------

class TestIndexScene:
    def test_returns_deterministic_id(self, indexer):
        record = _make_scene()
        pid = indexer.index_scene(record)
        assert pid == _scene_point_id("m1", "s1")

    def test_calls_upsert_on_scenes_collection(self, indexer, mock_client):
        record = _make_scene()
        indexer.index_scene(record)
        mock_client.upsert.assert_called_once()
        kwargs = mock_client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "scenes"

    def test_upserted_payload_contains_required_fields(self, indexer, mock_client):
        record = _make_scene(movie_id="m42", scene_id="sc7")
        indexer.index_scene(record)
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.payload["movie_id"] == "m42"
        assert point.payload["scene_id"] == "sc7"
        assert point.payload["scene_title"] == "INT. OFFICE - DAY"
        assert point.payload["character_names"] == ["ALICE"]

    def test_upserted_vector_matches_embedding(self, indexer, mock_client):
        record = _make_scene()
        indexer.index_scene(record)
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.vector == _EMB

    def test_missing_optional_fields_default_gracefully(self, indexer, mock_client):
        record = SceneRecord(
            movie_id="m1", movie_title="T", scene_id="s1", scene_index=0,
            text="text", embedding=_EMB,
            scene_title=None, character_names=None,
        )
        indexer.index_scene(record)
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.payload["scene_title"] is None
        assert point.payload["character_names"] == []


# ---------------------------------------------------------------------------
# index_sentence
# ---------------------------------------------------------------------------

class TestIndexSentence:
    def test_returns_deterministic_id(self, indexer):
        record = _make_sentence(position=10)
        pid = indexer.index_sentence(record)
        assert pid == _sentence_point_id("m1", "s1", 10)

    def test_calls_upsert_on_sentences_collection(self, indexer, mock_client):
        record = _make_sentence()
        indexer.index_sentence(record)
        mock_client.upsert.assert_called_once()
        kwargs = mock_client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "sentences"

    def test_payload_contains_line_type_and_character(self, indexer, mock_client):
        record = _make_sentence()
        indexer.index_sentence(record)
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.payload["line_type"] == "dialogue"
        assert point.payload["character_name"] == "ALICE"

    def test_description_line_has_no_character(self, indexer, mock_client):
        record = SentenceRecord(
            movie_id="m1", movie_title="T", scene_id="s1", scene_index=0,
            text="A dark corridor.", line_type="description",
            position_in_script=3, embedding=_EMB, character_name=None,
        )
        indexer.index_sentence(record)
        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.payload["character_name"] is None
        assert point.payload["line_type"] == "description"


# ---------------------------------------------------------------------------
# index_movie_batch
# ---------------------------------------------------------------------------

class TestIndexMovieBatch:
    def test_makes_exactly_two_upsert_calls(self, indexer, mock_client):
        scenes = [_make_scene(scene_id=f"s{i}") for i in range(3)]
        sentences = [_make_sentence(position=i) for i in range(5)]
        indexer.index_movie_batch(scenes, sentences)
        assert mock_client.upsert.call_count == 2

    def test_first_call_targets_scenes_collection(self, indexer, mock_client):
        indexer.index_movie_batch([_make_scene()], [_make_sentence()])
        first_call_kwargs = mock_client.upsert.call_args_list[0].kwargs
        assert first_call_kwargs["collection_name"] == "scenes"

    def test_second_call_targets_sentences_collection(self, indexer, mock_client):
        indexer.index_movie_batch([_make_scene()], [_make_sentence()])
        second_call_kwargs = mock_client.upsert.call_args_list[1].kwargs
        assert second_call_kwargs["collection_name"] == "sentences"

    def test_scene_batch_size_matches_input(self, indexer, mock_client):
        scenes = [_make_scene(scene_id=f"s{i}") for i in range(4)]
        indexer.index_movie_batch(scenes, [])
        points = mock_client.upsert.call_args_list[0].kwargs["points"]
        assert len(points) == 4

    def test_sentence_batch_size_matches_input(self, indexer, mock_client):
        sentences = [_make_sentence(position=i) for i in range(7)]
        indexer.index_movie_batch([], sentences)
        points = mock_client.upsert.call_args_list[0].kwargs["points"]
        assert len(points) == 7

    def test_empty_scenes_skips_scene_upsert(self, indexer, mock_client):
        indexer.index_movie_batch([], [_make_sentence()])
        assert mock_client.upsert.call_count == 1
        kwargs = mock_client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "sentences"

    def test_empty_sentences_skips_sentence_upsert(self, indexer, mock_client):
        indexer.index_movie_batch([_make_scene()], [])
        assert mock_client.upsert.call_count == 1
        kwargs = mock_client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "scenes"

    def test_scene_point_ids_are_deterministic(self, indexer, mock_client):
        scenes = [_make_scene(scene_id="fixed")]
        indexer.index_movie_batch(scenes, [])
        point = mock_client.upsert.call_args_list[0].kwargs["points"][0]
        assert point.id == _scene_point_id("m1", "fixed")


# ---------------------------------------------------------------------------
# index_movie convenience wrapper
# ---------------------------------------------------------------------------

class TestIndexMovieWrapper:
    def test_delegates_to_indexer(self, mock_client):
        with patch("vector_db.indexer.get_qdrant_client", return_value=mock_client):
            index_movie([_make_scene()], [_make_sentence()], config=_CFG)
        assert mock_client.upsert.call_count == 2
