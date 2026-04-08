"""Tests for the standalone raw-query cleaner script."""

from __future__ import annotations

import json
from pathlib import Path

from finetuning.corpus.scene_corpus import MovieEntry
from finetuning.dataset.dataset_builder import _read_jsonl as read_raw_query_jsonl
from finetuning.scripts import clean_raw_queries as cleaner
from preprocessing.chunker import SceneChunk


def _make_scene(
    movie_id: str,
    movie_title: str,
    scene_index: int,
    text: str,
    *,
    character_names: list[str] | None = None,
) -> SceneChunk:
    return SceneChunk(
        movie_id=movie_id,
        movie_title=movie_title,
        scene_id=f"scene_{scene_index:04d}",
        scene_index=scene_index,
        text=text,
        scene_title=f"SCENE {scene_index}",
        character_names=character_names or [],
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_repair_query_parses_json_wrapper() -> None:
    outcome = cleaner.repair_query('{"query": "A married teacher propositions a student", "skip": false}')
    assert outcome.cleaned_query == "A married teacher propositions a student"
    assert outcome.repair_method == "parsed_json_wrapper"
    assert "had_json_wrapper" in outcome.flags


def test_repair_query_parses_fenced_json() -> None:
    outcome = cleaner.repair_query('```json\n{"query": "A woman discovers a hidden affair"}\n```')
    assert outcome.cleaned_query == "A woman discovers a hidden affair"
    assert outcome.repair_method == "parsed_fenced_json_wrapper"
    assert "markdown_fence" in outcome.flags


def test_repair_query_salvages_partial_json_wrapper() -> None:
    outcome = cleaner.repair_query('{"query": "A woman discovers her husband has been living a double life')
    assert outcome.cleaned_query == "A woman discovers her husband has been living a double life"
    assert outcome.repair_method == "salvaged_partial_json"
    assert "partial_json_salvaged" in outcome.flags


def test_repair_query_rejects_truncated_plain_text() -> None:
    outcome = cleaner.repair_query("A woman discovers the truth about her father and")
    assert outcome.cleaned_query is None
    assert outcome.structural_reason == "likely_truncated"


def test_repair_query_keeps_clean_plain_text() -> None:
    outcome = cleaner.repair_query("A teenager fakes a severe illness with a dramatic seizure to avoid school")
    assert outcome.cleaned_query == "A teenager fakes a severe illness with a dramatic seizure to avoid school"
    assert outcome.repair_method == "as_is"

def test_score_query_flags_bad_patterns() -> None:
    scene = _make_scene(
        "secret_window",
        "Secret Window",
        0,
        "A long confrontation scene",
        character_names=["JOHN"],
    )

    synopsis_scored = cleaner.score_query(
        "Scenes of unlikely allies forced to collaborate",
        movie_title="Any Movie",
        query_type="synopsis",
    )
    assert "banned_synopsis_prefix" in synopsis_scored.flags
    assert "abstract_theme_wording" in synopsis_scored.flags

    title_scored = cleaner.score_query(
        "A woman uncovers the hidden truth behind Secret Window",
        movie_title="Secret Window",
        query_type="scene_summary",
        scene=scene,
    )
    assert "title_leakage" in title_scored.flags

    character_scored = cleaner.score_query(
        "John confesses to the murder during a tense courtroom confrontation",
        movie_title="Secret Window",
        query_type="scene_summary",
        scene=scene,
    )
    assert "character_name_leakage" in character_scored.flags

    generic_scored = cleaner.score_query(
        "A woman struggles with her life",
        movie_title="Any Movie",
        query_type="synopsis",
    )
    assert "generic_low_information" in generic_scored.flags

    short_scored = cleaner.score_query(
        "A dangerous secret emerges",
        movie_title="Any Movie",
        query_type="synopsis",
    )
    assert "soft_too_short" in short_scored.flags

    long_scored = cleaner.score_query(
        "A detective returns home after years away and uncovers a conspiracy involving corrupt officials, family betrayals, hidden money, a murder cover-up, a blackmail ring, and a decades-old disappearance that reshapes everything he believed about his family",
        movie_title="Any Movie",
        query_type="synopsis",
    )
    assert "too_long" in long_scored.flags


def test_run_cleaner_smoke_regeneration_and_outputs(monkeypatch, tmp_path: Path) -> None:
    movie_one_scene_short = _make_scene(
        "movie_one",
        "Movie One",
        0,
        "John stares at old photographs in silence.",
        character_names=["JOHN"],
    )
    movie_one_scene_long = _make_scene(
        "movie_one",
        "Movie One",
        1,
        "A wife discovers her husband has been living a double life after finding hidden passports and confronting him in their kitchen.",
        character_names=["ANNA", "MARK"],
    )
    movie_two_scene = _make_scene(
        "movie_two",
        "Secret Window",
        0,
        "John admits his crime during a heated confrontation in front of everyone at the cabin.",
        character_names=["JOHN"],
    )

    corpus = {
        "movie_one": MovieEntry(
            movie_id="movie_one",
            movie_title="Movie One",
            overview="Two strangers uncover a conspiracy while pretending to cooperate.",
            scenes=[movie_one_scene_short, movie_one_scene_long],
        ),
        "movie_two": MovieEntry(
            movie_id="movie_two",
            movie_title="Secret Window",
            overview="A writer unravels while confronting accusations and buried guilt.",
            scenes=[movie_two_scene],
        ),
    }

    class FakeQueryGenerator:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key

        def generate_scene_summary(self, scene_text: str, movie_title: str) -> str | None:
            if "double life" in scene_text:
                return "A wife discovers her husband has been living a double life and confronts him at home"
            return "John in Secret Window"

        def generate_synopsis_queries(self, overview: str, movie_title: str, n: int = 4) -> list[str]:
            return [
                "A pair of reluctant allies uncover a conspiracy while pretending to work together",
                "Two strangers fake cooperation while exposing a larger conspiracy",
                "Scenes of unlikely allies forced to collaborate",
            ][:n]

    monkeypatch.setattr(cleaner, "build_scene_corpus", lambda: corpus)
    monkeypatch.setattr(cleaner, "QueryGenerator", FakeQueryGenerator)

    input_rows = [
        {
            "movie_id": "movie_one",
            "movie_title": "Movie One",
            "query": "A teenager fakes a severe illness with a dramatic seizure to avoid school",
            "query_type": "synopsis",
            "scene_idx": None,
        },
        {
            "movie_id": "movie_one",
            "movie_title": "Movie One",
            "query": '{"query": "A wife discovers',
            "query_type": "scene_summary",
            "scene_idx": 0,
        },
        {
            "movie_id": "movie_one",
            "movie_title": "Movie One",
            "query": "Scenes of unlikely allies forced to collaborate",
            "query_type": "synopsis",
            "scene_idx": None,
        },
        {
            "movie_id": "movie_two",
            "movie_title": "Secret Window",
            "query": '{"query": "badly broken scene query',
            "query_type": "scene_summary",
            "scene_idx": 0,
        },
    ]

    input_path = tmp_path / "raw_queries.jsonl"
    output_path = tmp_path / "raw_queries.cleaned.jsonl"
    audit_path = tmp_path / "raw_queries.audit.jsonl"
    rejected_path = tmp_path / "raw_queries.rejected.jsonl"
    report_path = tmp_path / "raw_queries.cleaning_report.json"
    _write_jsonl(input_path, input_rows)

    exit_code = cleaner.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--audit",
            str(audit_path),
            "--rejected",
            str(rejected_path),
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0

    cleaned_rows = read_raw_query_jsonl(output_path)
    assert all(set(row.keys()) == {"movie_id", "movie_title", "query", "query_type", "scene_idx"} for row in cleaned_rows)
    assert any(row["query"] == "A wife discovers her husband has been living a double life and confronts him at home" for row in cleaned_rows)
    assert any(row["query"] == "A pair of reluctant allies uncover a conspiracy while pretending to work together" for row in cleaned_rows)
    assert not any(row["query"] == "Scenes of unlikely allies forced to collaborate" for row in cleaned_rows)

    rejected_rows = read_raw_query_jsonl(rejected_path)
    assert any(row["reject_reason"] == "regenerated_scene_failed_quality" for row in rejected_rows)

    with open(audit_path, "r", encoding="utf-8") as f:
        audit_rows = [json.loads(line) for line in f if line.strip()]
    assert any(row["action"] == "regenerate" and row["was_regenerated"] for row in audit_rows)
    assert any(row["action"] == "reject" for row in audit_rows)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["action_counts"]["keep"] >= 1
    assert report["action_counts"]["regenerate"] >= 1
    assert report["action_counts"]["reject"] >= 1
