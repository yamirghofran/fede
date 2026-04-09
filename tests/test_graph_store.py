"""Tests for knowledge_graph.graph_store."""

from pathlib import Path

from knowledge_graph.graph_models import PatternQueryRequest
from knowledge_graph.graph_store import KnowledgeGraphService, MemoryGraphBackend


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")


def _seed_movie(entities_dir: Path, relations_dir: Path, movie_id: str, entities: list[dict], relations: list[dict]) -> None:
    _write_json(entities_dir / f"{movie_id}_entities.json", {"file": f"{movie_id}.txt", "entities": entities})
    _write_json(relations_dir / f"{movie_id}_relations.json", {"file": f"{movie_id}.txt", "relations": relations})


def test_build_and_reload_movie(tmp_path: Path):
    entities_dir = tmp_path / "entities_clean"
    relations_dir = tmp_path / "relations"
    entities_dir.mkdir()
    relations_dir.mkdir()
    _seed_movie(
        entities_dir,
        relations_dir,
        "movie-a",
        [
            {"text": "Alice", "label": "PERSON"},
            {"text": "Bob", "label": "PERSON"},
            {"text": "Carol", "label": "PERSON"},
        ],
        [
            {
                "from": "Alice",
                "from_type": "PERSON",
                "to": "Bob",
                "to_type": "PERSON",
                "label": "TEACHES",
                "evidence": "Alice trains Bob.",
            },
            {
                "from": "Bob",
                "from_type": "PERSON",
                "to": "Carol",
                "to_type": "PERSON",
                "label": "BETRAYS",
                "evidence": "Bob betrays Carol.",
            },
        ],
    )
    _seed_movie(
        entities_dir,
        relations_dir,
        "movie-b",
        [
            {"text": "Dana", "label": "PERSON"},
            {"text": "Eve", "label": "PERSON"},
        ],
        [
            {
                "from": "Dana",
                "from_type": "PERSON",
                "to": "Eve",
                "to_type": "PERSON",
                "label": "SAVES",
                "evidence": "Dana saves Eve.",
            }
        ],
    )
    service = KnowledgeGraphService(
        db_path=tmp_path / "story_graph.db",
        entities_dir=entities_dir,
        relations_dir=relations_dir,
        backend=MemoryGraphBackend(),
    )

    summary = service.build(rebuild=True)
    health = service.health()

    assert summary.movies_loaded == 2
    assert health.counts.movies == 2
    assert health.counts.entities == 5
    assert health.counts.narrative_edges == 3

    _seed_movie(
        entities_dir,
        relations_dir,
        "movie-b",
        [
            {"text": "Dana", "label": "PERSON"},
            {"text": "Eve", "label": "PERSON"},
            {"text": "Finn", "label": "PERSON"},
        ],
        [
            {
                "from": "Dana",
                "from_type": "PERSON",
                "to": "Eve",
                "to_type": "PERSON",
                "label": "SAVES",
                "evidence": "Dana saves Eve.",
            },
            {
                "from": "Eve",
                "from_type": "PERSON",
                "to": "Finn",
                "to_type": "PERSON",
                "label": "TEACHES",
                "evidence": "Eve teaches Finn.",
            },
        ],
    )
    reload_summary = service.build(movie_id="movie-b")
    reloaded_health = service.health()

    assert reload_summary.mode == "movie_reload"
    assert reload_summary.movies_loaded == 1
    assert reloaded_health.counts.movies == 2
    assert reloaded_health.counts.entities == 6
    assert reloaded_health.counts.narrative_edges == 4


def test_pattern_query_finds_teaches_then_betrays(tmp_path: Path):
    entities_dir = tmp_path / "entities_clean"
    relations_dir = tmp_path / "relations"
    entities_dir.mkdir()
    relations_dir.mkdir()
    _seed_movie(
        entities_dir,
        relations_dir,
        "movie-a",
        [
            {"text": "Alice", "label": "PERSON"},
            {"text": "Bob", "label": "PERSON"},
            {"text": "Carol", "label": "PERSON"},
        ],
        [
            {
                "from": "Alice",
                "from_type": "PERSON",
                "to": "Bob",
                "to_type": "PERSON",
                "label": "TEACHES",
                "evidence": "Alice trains Bob.",
            },
            {
                "from": "Bob",
                "from_type": "PERSON",
                "to": "Carol",
                "to_type": "PERSON",
                "label": "BETRAYS",
                "evidence": "Bob betrays Carol.",
            },
        ],
    )
    service = KnowledgeGraphService(
        db_path=tmp_path / "story_graph.db",
        entities_dir=entities_dir,
        relations_dir=relations_dir,
        backend=MemoryGraphBackend(),
    )
    service.build(rebuild=True)

    response = service.query_pattern(PatternQueryRequest(predicates=["TEACHES", "BETRAYS"], limit=5))

    assert len(response.results) == 1
    assert response.results[0].movie_id == "movie-a"
    assert [step.entity_name for step in response.results[0].path] == ["Alice", "Bob", "Carol"]
