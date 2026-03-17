"""Tests for vector_db.schemas."""

import pytest

from vector_db.schemas import CollectionNames, ScenePayload, SentencePayload


class TestCollectionNames:
    def test_values_are_strings(self):
        assert CollectionNames.SCENES.value == "scenes"
        assert CollectionNames.SENTENCES.value == "sentences"

    def test_enum_is_str_subclass(self):
        # CollectionNames inherits str so values can be passed directly as
        # Qdrant collection name strings without calling .value.
        assert isinstance(CollectionNames.SCENES, str)
        assert CollectionNames.SCENES == "scenes"
        assert CollectionNames.SENTENCES == "sentences"

    def test_all_expected_members_present(self):
        names = {m.value for m in CollectionNames}
        assert names == {"scenes", "sentences"}


class TestSentencePayload:
    def test_full_dialogue_payload(self):
        payload: SentencePayload = {
            "movie_id": "tt0111161",
            "movie_title": "The Shawshank Redemption",
            "scene_id": "scene-001",
            "scene_index": 0,
            "text": "Get busy living, or get busy dying.",
            "line_type": "dialogue",
            "character_name": "RED",
            "position_in_script": 42,
        }
        assert payload["line_type"] == "dialogue"
        assert payload["character_name"] == "RED"

    def test_description_payload_no_character(self):
        payload: SentencePayload = {
            "movie_id": "tt0111161",
            "movie_title": "The Shawshank Redemption",
            "scene_id": "scene-002",
            "scene_index": 1,
            "text": "A two-lane road carves through an endless forest.",
            "line_type": "description",
            "character_name": None,
            "position_in_script": 10,
        }
        assert payload["line_type"] == "description"
        assert payload["character_name"] is None

    def test_partial_payload_allowed(self):
        # total=False means TypedDict allows partial construction.
        payload: SentencePayload = {"movie_id": "tt0111161", "text": "Hello."}  # type: ignore[typeddict-item]
        assert payload["movie_id"] == "tt0111161"


class TestScenePayload:
    def test_full_scene_payload(self):
        payload: ScenePayload = {
            "movie_id": "tt0111161",
            "movie_title": "The Shawshank Redemption",
            "scene_id": "scene-001",
            "scene_index": 0,
            "text": "INT. PRISON CELL - NIGHT\nAndy sits alone.",
            "scene_title": "INT. PRISON CELL - NIGHT",
            "character_names": ["ANDY", "GUARD"],
        }
        assert payload["scene_title"] == "INT. PRISON CELL - NIGHT"
        assert "ANDY" in payload["character_names"]

    def test_scene_without_title(self):
        payload: ScenePayload = {
            "movie_id": "tt0111161",
            "movie_title": "The Shawshank Redemption",
            "scene_id": "scene-002",
            "scene_index": 1,
            "text": "The yard is empty.",
            "scene_title": None,
            "character_names": [],
        }
        assert payload["scene_title"] is None
        assert payload["character_names"] == []
