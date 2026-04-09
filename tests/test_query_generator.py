"""Tests for shared query-generation parsing and synopsis robustness."""

from __future__ import annotations

from finetuning.dataset import query_generator as qg


def test_parse_json_string_list_salvages_partial_queries_array() -> None:
    raw = (
        '{"queries": ['
        '"Dark wizard leader recruiting followers for a supremacist cause.", '
        '"A powerful wizard confronting former allies", '
        '"A character realiz'
    )
    parsed = qg._parse_json_string_list(raw, preferred_key="queries")
    assert parsed == [
        "Dark wizard leader recruiting followers for a supremacist cause.",
        "A powerful wizard confronting former allies",
    ]


def test_parse_json_string_list_salvages_partial_paraphrases_array() -> None:
    raw = (
        '{"paraphrases": ['
        '"A mother searches for the son taken from her years ago.", '
        '"A woman tracks down the child she was forced to give up", '
        '"A parent recon'
    )
    parsed = qg._parse_json_string_list(raw, preferred_key="paraphrases")
    assert parsed == [
        "A mother searches for the son taken from her years ago.",
        "A woman tracks down the child she was forced to give up",
    ]


def test_generate_synopsis_queries_chunks_requests_and_salvages_partial_json() -> None:
    class FakeSynopsisGenerator:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.call_count = 0

        def _call_llm(self, prompt: str, *, response_format: dict | None = None) -> str | None:
            self.call_count += 1
            if "write 4 distinct short queries" in prompt:
                self.calls.append(4)
                if self.call_count == 1:
                    return (
                        '{"queries": ['
                        '"Dark wizard leader recruiting followers for a supremacist cause.", '
                        '"A powerful wizard confronting former allies", '
                        '"A character realiz'
                    )
                return (
                    '{"queries": ['
                    '"A dangerous alliance forms around a radical wizard movement", '
                    '"Former allies clash over a wizard supremacist plan", '
                    '"A secret mission tries to stop a rising magical extremist", '
                    '"Conflicting loyalties split a group trying to stop a radical leader"'
                    ']}'
                )
            raise AssertionError(f"Unexpected prompt: {prompt}")

    fake = FakeSynopsisGenerator()
    queries = qg.QueryGenerator.generate_synopsis_queries(
        fake,  # type: ignore[arg-type]
        overview="A dark wizard gathers followers while former allies try to stop him.",
        movie_title="Fantastic Beasts: The Crimes of Grindelwald",
        n=6,
    )

    assert fake.calls == [4, 4]
    assert len(queries) == 6
    assert "Dark wizard leader recruiting followers for a supremacist cause." in queries
    assert "A powerful wizard confronting former allies" in queries
