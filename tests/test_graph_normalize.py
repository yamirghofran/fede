"""Tests for knowledge_graph.graph_normalize."""

from pathlib import Path

from knowledge_graph.graph_normalize import load_movie_document, stable_entity_id, stable_relation_id


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")


def test_valid_relation_survives_and_ids_are_stable(tmp_path: Path):
    entities_dir = tmp_path / "entities_clean"
    relations_dir = tmp_path / "relations"
    entities_dir.mkdir()
    relations_dir.mkdir()
    _write_json(
        entities_dir / "sample_entities.json",
        {
            "file": "sample.txt",
            "entities": [
                {"text": "Alice", "label": "PERSON"},
                {"text": "Bob", "label": "PERSON"},
            ],
        },
    )
    _write_json(
        relations_dir / "sample_relations.json",
        {
            "file": "sample.txt",
            "relations": [
                {
                    "from": "Alice",
                    "from_type": "PERSON",
                    "to": "Bob",
                    "to_type": "PERSON",
                    "label": "TEACHES",
                    "evidence": "Alice teaches Bob to fight.",
                }
            ],
        },
    )

    document = load_movie_document("sample", entities_dir, relations_dir)

    assert len(document.entities) == 2
    assert len(document.relations) == 1
    relation = document.relations[0]
    assert relation.from_name == "Alice"
    assert relation.to_name == "Bob"
    assert relation.relation_id == stable_relation_id(
        "sample",
        stable_entity_id("sample", "PERSON", "Alice"),
        "TEACHES",
        stable_entity_id("sample", "PERSON", "Bob"),
        "Alice teaches Bob to fight.",
    )


def test_malformed_relation_is_dropped_with_reason(tmp_path: Path):
    entities_dir = tmp_path / "entities_clean"
    relations_dir = tmp_path / "relations"
    entities_dir.mkdir()
    relations_dir.mkdir()
    _write_json(
        entities_dir / "sample_entities.json",
        {
            "file": "sample.txt",
            "entities": [
                {"text": "Alice", "label": "PERSON"},
            ],
        },
    )
    _write_json(
        relations_dir / "sample_relations.json",
        {
            "file": "sample.txt",
            "relations": [
                {
                    "from": "Alice",
                    "from_type": "PERSON",
                    "to": "Unknown",
                    "to_type": "PERSON",
                    "label": "TEACHES",
                    "evidence": "Alice teaches someone.",
                }
            ],
        },
    )

    document = load_movie_document("sample", entities_dir, relations_dir)

    assert not document.relations
    assert [item.reason for item in document.dropped_relations] == ["missing_to_entity"]


def test_duplicate_relations_are_deduplicated(tmp_path: Path):
    entities_dir = tmp_path / "entities_clean"
    relations_dir = tmp_path / "relations"
    entities_dir.mkdir()
    relations_dir.mkdir()
    _write_json(
        entities_dir / "sample_entities.json",
        {
            "file": "sample.txt",
            "entities": [
                {"text": "Alice", "label": "PERSON"},
                {"text": "Bob", "label": "PERSON"},
            ],
        },
    )
    relation = {
        "from": "Alice",
        "from_type": "PERSON",
        "to": "Bob",
        "to_type": "PERSON",
        "label": "TEACHES",
        "evidence": "Alice teaches Bob to fight.",
    }
    _write_json(
        relations_dir / "sample_relations.json",
        {"file": "sample.txt", "relations": [relation, dict(relation)]},
    )

    document = load_movie_document("sample", entities_dir, relations_dir)

    assert len(document.relations) == 1


def test_noisy_target_is_excluded(tmp_path: Path):
    entities_dir = tmp_path / "entities_clean"
    relations_dir = tmp_path / "relations"
    entities_dir.mkdir()
    relations_dir.mkdir()
    _write_json(
        entities_dir / "whiplash_entities.json",
        {
            "file": "whiplash.txt",
            "entities": [
                {"text": "Fletcher", "label": "PERSON"},
                {"text": "Andrew Neiman", "label": "PERSON"},
            ],
        },
    )
    _write_json(
        relations_dir / "whiplash_relations.json",
        {
            "file": "whiplash.txt",
            "relations": [
                {
                    "from": "Fletcher",
                    "from_type": "PERSON",
                    "to": "Andrew BEGINS",
                    "to_type": "PERSON",
                    "label": "TEACHES",
                    "evidence": "You still have other options.",
                }
            ],
        },
    )

    document = load_movie_document("whiplash", entities_dir, relations_dir)

    assert not document.relations
    assert [item.reason for item in document.dropped_relations] == ["missing_to_entity"]
