#!/usr/bin/env python3
"""Standalone cleaner for raw query JSONL files.

This script repairs malformed raw queries, scores query quality, optionally
regenerates weak rows from their source scene/synopsis using the existing LLM
stack, deduplicates the results, and writes schema-compatible output for the
current pair-assembly pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from finetuning.config import FINETUNING_DATA_DIR, TOP_SCENES_FOR_SUMMARY
from finetuning.corpus.scene_corpus import MovieEntry, build_scene_corpus
from finetuning.dataset.query_generator import QueryGenerator, check_leakage
from preprocessing.chunker import SceneChunk

logger = logging.getLogger(__name__)

_DEFAULT_INPUT = FINETUNING_DATA_DIR / "raw_queries.jsonl"
_DEFAULT_OUTPUT = FINETUNING_DATA_DIR / "raw_queries.cleaned.jsonl"
_DEFAULT_AUDIT = FINETUNING_DATA_DIR / "raw_queries.audit.jsonl"
_DEFAULT_REJECTED = FINETUNING_DATA_DIR / "raw_queries.rejected.jsonl"
_DEFAULT_REPORT = FINETUNING_DATA_DIR / "raw_queries.cleaning_report.json"

_SYNOPSIS_META_PREFIXES = (
    "scenes of",
    "looking for",
    "film scenes",
    "a movie where",
    "a film about",
)

_GENERIC_SUBJECT_PATTERNS = (
    "a man",
    "a woman",
    "someone",
    "somebody",
    "a family",
    "friends",
    "a couple",
    "a group",
)

_DISTINCTIVE_EVENT_TERMS = (
    "betrayal",
    "betrays",
    "confession",
    "confesses",
    "courtroom",
    "testimony",
    "trial",
    "fake illness",
    "seizure",
    "kidnapping",
    "kidnapped",
    "double life",
    "abuse",
    "mistaken identity",
    "blackmail",
    "escape",
    "escapes",
    "arrest",
    "arrested",
    "accident",
    "discovers",
    "discovery",
    "confrontation",
    "confronts",
    "proposal",
    "proposes",
    "reunion",
    "reunites",
    "death reveal",
    "reveals",
    "secret",
    "murder",
    "affair",
    "hostage",
)

_ABSTRACT_TERMS = (
    "exploring",
    "explores",
    "dynamics",
    "unlikely allies",
    "strained family",
    "identity",
    "redemption",
    "theme",
    "themes",
    "relationship dynamic",
    "family dynamics",
)

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "that", "this", "it", "its",
    "he", "she", "they", "we", "i", "you", "as", "if", "not", "no", "so",
    "up", "out", "about", "into", "than", "then", "when", "who", "which",
    "what", "how", "where", "why", "their", "his", "her", "after", "before",
    "during", "while", "through", "across", "under", "over",
}

_TRUNCATION_ENDINGS = {
    "and", "or", "but", "because", "when", "while", "with", "after", "before",
    "into", "from", "than", "then", "where", "who", "whose", "which",
}

_MIN_WORDS = 5
_SOFT_MIN_WORDS = 7
_SCENE_MAX_WORDS = 35
_SYNOPSIS_MAX_WORDS = 28
_SCENE_KEEP_THRESHOLD = 0.75
_SYNOPSIS_KEEP_THRESHOLD = 0.85
_NEAR_DUPLICATE_THRESHOLD = 0.85


@dataclass
class RepairOutcome:
    cleaned_query: Optional[str]
    repair_method: str
    flags: List[str] = field(default_factory=list)
    structural_reason: Optional[str] = None


@dataclass
class ScoredQuery:
    query: str
    score: float
    flags: List[str]


@dataclass
class RowState:
    movie_id: str
    movie_title: str
    query_type: str
    scene_idx: Optional[int]
    original_query: str
    cleaned_query: Optional[str]
    repair_method: str
    flags: List[str] = field(default_factory=list)
    reject_reason: Optional[str] = None
    quality_score: float = 0.0
    action: str = "reject"
    was_regenerated: bool = False
    scene: Optional[SceneChunk] = None
    overview: Optional[str] = None
    source_missing: bool = False


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw FEDE training queries")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Input raw_queries JSONL")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Cleaned raw_queries JSONL")
    parser.add_argument("--audit", type=Path, default=_DEFAULT_AUDIT, help="Audit JSONL path")
    parser.add_argument("--rejected", type=Path, default=_DEFAULT_REJECTED, help="Rejected rows JSONL path")
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT, help="Cleaning report JSON path")
    parser.add_argument("--movies", type=int, default=None, help="Process only the first N movies from the input JSONL")
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key override for LLM regeneration")
    parser.add_argument("--no-regenerate", action="store_true", help="Disable LLM regeneration and only keep deterministic-clean rows")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _strip_markdown_fences(text: str) -> Tuple[str, bool]:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped, False
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip(), True


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_query_text(text: str) -> str:
    return _normalize_whitespace(_normalize_quotes(text))


def _count_words(text: str) -> int:
    return len(text.split())


def _word_limit(query_type: str) -> int:
    return _SCENE_MAX_WORDS if query_type == "scene_summary" else _SYNOPSIS_MAX_WORDS


def _keep_threshold(query_type: str) -> float:
    return _SCENE_KEEP_THRESHOLD if query_type == "scene_summary" else _SYNOPSIS_KEEP_THRESHOLD


def _tokenize_normalized(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if tok]


def _content_tokens(text: str) -> List[str]:
    return [tok for tok in _tokenize_normalized(text) if tok not in _STOPWORDS]


def _normalized_key(text: str) -> str:
    return " ".join(_content_tokens(text))


def _jaccard_similarity(a: str, b: str) -> float:
    aset = set(_content_tokens(a))
    bset = set(_content_tokens(b))
    if not aset and not bset:
        return 1.0
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / len(aset | bset)


def _has_wrapper_artifacts(text: str) -> bool:
    lowered = text.lower()
    return (
        lowered.startswith("{")
        or '"query":' in lowered
        or lowered.startswith('"query":')
        or lowered.startswith("query:")
    )


def _looks_truncated(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.count("{") != stripped.count("}"):
        return True
    if stripped.count('"') % 2 == 1:
        return True
    if re.search(r"(\\n|\\t|\\r)", stripped):
        return True
    last_token = re.findall(r"[A-Za-z]+", stripped.lower())
    if last_token and last_token[-1] in _TRUNCATION_ENDINGS:
        return True
    if stripped.endswith(("...", "…")):
        return True
    return False


def _parse_wrapped_json(text: str) -> Optional[str]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        if parsed.get("skip", False) or parsed.get("is_usable", True) is False:
            return None
        query = parsed.get("query")
        if isinstance(query, str):
            return _normalize_query_text(query)
        return None
    if isinstance(parsed, str):
        return _normalize_query_text(parsed)
    return None


def _salvage_partial_json_query(text: str) -> Optional[str]:
    match = re.search(r'"query"\s*:\s*"', text, re.IGNORECASE)
    if not match:
        return None
    remainder = text[match.end():]
    remainder = remainder.replace('\\"', '"')
    if '"' in remainder:
        candidate = remainder.split('"', 1)[0]
    else:
        candidate = remainder.rstrip(" ,}")
    candidate = _normalize_query_text(candidate)
    if not candidate or _count_words(candidate) < _MIN_WORDS:
        return None
    if "{" in candidate or "}" in candidate or '"query":' in candidate.lower():
        return None
    return candidate


def repair_query(raw_query: Any) -> RepairOutcome:
    text = "" if raw_query is None else str(raw_query)
    text = _normalize_quotes(text).strip()
    if not text:
        return RepairOutcome(
            cleaned_query=None,
            repair_method="empty",
            structural_reason="empty_query",
        )

    stripped, had_fence = _strip_markdown_fences(text)
    flags: List[str] = []
    repair_method = "as_is"
    if had_fence:
        flags.append("markdown_fence")
        repair_method = "stripped_markdown_fence"

    parsed = _parse_wrapped_json(stripped)
    if parsed is not None:
        flags.append("had_json_wrapper")
        return RepairOutcome(
            cleaned_query=parsed,
            repair_method="parsed_json_wrapper" if not had_fence else "parsed_fenced_json_wrapper",
            flags=flags,
        )

    salvaged = _salvage_partial_json_query(stripped)
    if salvaged is not None:
        flags.extend(["had_json_wrapper", "partial_json_salvaged"])
        return RepairOutcome(
            cleaned_query=salvaged,
            repair_method="salvaged_partial_json",
            flags=flags,
        )

    normalized = _normalize_query_text(stripped.strip('"').strip("'"))
    if _has_wrapper_artifacts(normalized):
        return RepairOutcome(
            cleaned_query=None,
            repair_method=repair_method,
            flags=flags + ["json_wrapper_artifact"],
            structural_reason="json_wrapper_artifact",
        )

    if _looks_truncated(normalized):
        return RepairOutcome(
            cleaned_query=None,
            repair_method=repair_method,
            flags=flags + ["likely_truncated"],
            structural_reason="likely_truncated",
        )

    if _count_words(normalized) < _MIN_WORDS:
        return RepairOutcome(
            cleaned_query=None,
            repair_method=repair_method,
            flags=flags + ["too_short"],
            structural_reason="too_short",
        )

    if normalized != text:
        repair_method = "normalized_plain_text"
    return RepairOutcome(cleaned_query=normalized, repair_method=repair_method, flags=flags)


def _has_banned_synopsis_prefix(query: str) -> bool:
    lowered = query.lower()
    return any(lowered.startswith(prefix) for prefix in _SYNOPSIS_META_PREFIXES)


def _has_character_leakage(query: str, scene: Optional[SceneChunk]) -> bool:
    if scene and scene.character_names:
        lowered = query.lower()
        for name in scene.character_names:
            tokens = [tok for tok in re.findall(r"[a-z0-9]+", name.lower()) if len(tok) > 1]
            if tokens and all(tok in lowered for tok in tokens):
                return True
    tokens = re.findall(r"\b[A-Z][a-z]+\b", query)
    if len(tokens) > 1:
        return True
    return False


def _is_generic_low_information(query: str) -> bool:
    lowered = query.lower()
    has_generic_subject = any(pattern in lowered for pattern in _GENERIC_SUBJECT_PATTERNS)
    has_distinctive_event = any(term in lowered for term in _DISTINCTIVE_EVENT_TERMS)
    return has_generic_subject and not has_distinctive_event


def _has_abstract_theme_wording(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in _ABSTRACT_TERMS)


def score_query(
    query: str,
    *,
    movie_title: str,
    query_type: str,
    scene: Optional[SceneChunk] = None,
) -> ScoredQuery:
    score = 1.0
    flags: List[str] = []
    lowered = query.lower()
    word_count = _count_words(query)

    if _has_wrapper_artifacts(query):
        score -= 0.60
        flags.append("json_wrapper_artifact")
    if _looks_truncated(query):
        score -= 0.45
        flags.append("likely_truncated")
    if query_type == "synopsis" and _has_banned_synopsis_prefix(query):
        score -= 0.25
        flags.append("banned_synopsis_prefix")
    if check_leakage(query, movie_title):
        score -= 0.20
        flags.append("title_leakage")
    if _has_character_leakage(query, scene):
        score -= 0.20
        flags.append("character_name_leakage")
    if _is_generic_low_information(query):
        score -= 0.20
        flags.append("generic_low_information")
    if word_count < _SOFT_MIN_WORDS:
        score -= 0.15
        flags.append("soft_too_short")
    if word_count > _word_limit(query_type):
        score -= 0.15
        flags.append("too_long")
    if query.count(",") + query.count(";") + query.count(" and ") >= 3:
        score -= 0.15
        flags.append("overstuffed_clauses")
    if _has_abstract_theme_wording(query):
        score -= 0.10
        flags.append("abstract_theme_wording")
    content_tokens = _content_tokens(query)
    if len(content_tokens) < 4:
        score -= 0.10
        flags.append("low_content_density")
    if lowered.startswith("what happens when ") or lowered.startswith("why does "):
        score -= 0.10
        flags.append("question_meta_style")

    return ScoredQuery(query=query, score=max(0.0, round(score, 4)), flags=flags)


def _select_movie_rows(rows: List[Dict[str, Any]], max_movies: Optional[int]) -> List[Dict[str, Any]]:
    if max_movies is None:
        return rows
    selected_ids: List[str] = []
    seen: set[str] = set()
    for row in rows:
        movie_id = str(row.get("movie_id", ""))
        if movie_id and movie_id not in seen:
            seen.add(movie_id)
            selected_ids.append(movie_id)
            if len(selected_ids) >= max_movies:
                break
    allowed = set(selected_ids)
    return [row for row in rows if str(row.get("movie_id", "")) in allowed]


def _resolve_scene(entry: Optional[MovieEntry], scene_idx: Optional[int]) -> Optional[SceneChunk]:
    if entry is None or scene_idx is None:
        return None
    sorted_scenes = sorted(entry.scenes, key=lambda s: len(s.text), reverse=True)
    top_scenes = sorted_scenes[:TOP_SCENES_FOR_SUMMARY]
    if scene_idx < 0 or scene_idx >= len(top_scenes):
        return None
    return top_scenes[scene_idx]


def _can_regenerate(state: RowState) -> bool:
    if state.query_type == "scene_summary":
        return state.scene is not None
    return bool(state.overview and state.overview.strip())


def _initial_action(state: RowState, allow_regenerate: bool) -> str:
    if state.cleaned_query is None:
        if allow_regenerate and _can_regenerate(state):
            return "regenerate"
        return "reject"
    threshold = _keep_threshold(state.query_type)
    if state.quality_score >= threshold:
        return "keep"
    if allow_regenerate and _can_regenerate(state):
        return "regenerate"
    return "reject"


def _make_clean_output_row(state: RowState) -> Dict[str, Any]:
    return {
        "movie_id": state.movie_id,
        "movie_title": state.movie_title,
        "query": state.cleaned_query,
        "query_type": state.query_type,
        "scene_idx": state.scene_idx,
    }


def _make_audit_row(state: RowState) -> Dict[str, Any]:
    return {
        "movie_id": state.movie_id,
        "movie_title": state.movie_title,
        "query_type": state.query_type,
        "scene_idx": state.scene_idx,
        "original_query": state.original_query,
        "cleaned_query": state.cleaned_query,
        "action": state.action,
        "repair_method": state.repair_method,
        "quality_score": state.quality_score,
        "flags": state.flags,
        "reject_reason": state.reject_reason,
        "was_regenerated": state.was_regenerated,
    }


def _make_rejected_row(state: RowState) -> Dict[str, Any]:
    row = _make_audit_row(state)
    row["query"] = state.original_query
    return row


def _mark_rejected(state: RowState, reason: str) -> None:
    state.action = "reject"
    state.reject_reason = reason
    state.cleaned_query = state.cleaned_query if state.cleaned_query else None


def _is_duplicate_against(query: str, existing: Sequence[RowState]) -> bool:
    for candidate in existing:
        if not candidate.cleaned_query:
            continue
        if _normalized_key(candidate.cleaned_query) == _normalized_key(query):
            return True
        if _jaccard_similarity(candidate.cleaned_query, query) >= _NEAR_DUPLICATE_THRESHOLD:
            return True
    return False


def _rank_for_duplicate_preference(state: RowState) -> Tuple[float, int]:
    return (state.quality_score, 1 if not state.was_regenerated else 0)


def _deduplicate_kept_rows(states: List[RowState]) -> Tuple[List[RowState], Dict[str, int]]:
    kept = [state for state in states if state.action in {"keep", "regenerate"} and state.cleaned_query]
    duplicate_stats = {
        "pre_dedupe_rows": len(kept),
        "exact_duplicates_removed": 0,
        "near_duplicates_removed": 0,
    }

    by_movie: Dict[Tuple[str, str], List[RowState]] = defaultdict(list)
    for state in kept:
        by_movie[(state.movie_id, state.query_type)].append(state)

    for _, group in by_movie.items():
        survivors: List[RowState] = []
        for state in sorted(group, key=_rank_for_duplicate_preference, reverse=True):
            assert state.cleaned_query is not None
            exact = any(_normalized_key(s.cleaned_query or "") == _normalized_key(state.cleaned_query) for s in survivors)
            near = any(_jaccard_similarity(s.cleaned_query or "", state.cleaned_query) >= _NEAR_DUPLICATE_THRESHOLD for s in survivors)
            if exact or near:
                _mark_rejected(state, "duplicate_within_movie")
                if exact:
                    duplicate_stats["exact_duplicates_removed"] += 1
                else:
                    duplicate_stats["near_duplicates_removed"] += 1
                continue
            survivors.append(state)

    global_survivors: List[RowState] = []
    for state in sorted(
        [s for s in states if s.action in {"keep", "regenerate"} and s.cleaned_query],
        key=_rank_for_duplicate_preference,
        reverse=True,
    ):
        assert state.cleaned_query is not None
        exact = any(_normalized_key(s.cleaned_query or "") == _normalized_key(state.cleaned_query) for s in global_survivors)
        near = any(_jaccard_similarity(s.cleaned_query or "", state.cleaned_query) >= _NEAR_DUPLICATE_THRESHOLD for s in global_survivors)
        if exact or near:
            _mark_rejected(state, "duplicate_global")
            if exact:
                duplicate_stats["exact_duplicates_removed"] += 1
            else:
                duplicate_stats["near_duplicates_removed"] += 1
            continue
        global_survivors.append(state)

    duplicate_stats["post_dedupe_rows"] = len(global_survivors)
    duplicate_stats["duplicates_removed"] = duplicate_stats["pre_dedupe_rows"] - duplicate_stats["post_dedupe_rows"]
    return global_survivors, duplicate_stats


def _regenerate_scene_summary(state: RowState, qgen: QueryGenerator) -> None:
    if state.scene is None:
        _mark_rejected(state, "missing_source_scene")
        return
    regenerated = qgen.generate_scene_summary(state.scene.text, state.movie_title)
    if not regenerated:
        _mark_rejected(state, "scene_regeneration_failed")
        return
    scored = score_query(
        regenerated,
        movie_title=state.movie_title,
        query_type=state.query_type,
        scene=state.scene,
    )
    if scored.score < _keep_threshold(state.query_type):
        state.cleaned_query = regenerated
        state.quality_score = scored.score
        state.flags = sorted(set(state.flags + scored.flags + ["regenerated"]))
        _mark_rejected(state, "regenerated_scene_failed_quality")
        return
    state.cleaned_query = regenerated
    state.quality_score = scored.score
    state.flags = sorted(set(state.flags + scored.flags + ["regenerated"]))
    state.action = "regenerate"
    state.reject_reason = None
    state.was_regenerated = True


def _regenerate_synopsis_group(states: List[RowState], entry: MovieEntry, qgen: QueryGenerator, all_states: List[RowState]) -> None:
    if not entry.overview or not entry.overview.strip():
        for state in states:
            _mark_rejected(state, "missing_overview")
        return

    baseline = [
        candidate
        for candidate in all_states
        if candidate.movie_id == entry.movie_id
        and candidate.query_type == "synopsis"
        and candidate.action == "keep"
        and candidate.cleaned_query
    ]

    needed = len(states)
    pool_size = max(needed + 2, needed * 2)
    candidates: List[ScoredQuery] = []
    attempts = 0
    while len(candidates) < needed and attempts < 2:
        attempts += 1
        generated = qgen.generate_synopsis_queries(entry.overview, entry.movie_title, n=pool_size)
        for query in generated:
            scored = score_query(
                query,
                movie_title=entry.movie_title,
                query_type="synopsis",
            )
            if scored.score < _keep_threshold("synopsis"):
                continue
            if _is_duplicate_against(query, baseline):
                continue
            if any(_normalized_key(existing.query) == _normalized_key(query) for existing in candidates):
                continue
            if any(_jaccard_similarity(existing.query, query) >= _NEAR_DUPLICATE_THRESHOLD for existing in candidates):
                continue
            candidates.append(scored)
        pool_size = max(pool_size + 2, pool_size * 2)

    candidates.sort(key=lambda item: item.score, reverse=True)
    for state, candidate in zip(states, candidates):
        state.cleaned_query = candidate.query
        state.quality_score = candidate.score
        state.flags = sorted(set(state.flags + candidate.flags + ["regenerated"]))
        state.action = "regenerate"
        state.reject_reason = None
        state.was_regenerated = True
        baseline.append(state)

    for state in states[len(candidates):]:
        _mark_rejected(state, "insufficient_regenerated_synopsis")


def _build_report(states: List[RowState], duplicate_stats: Dict[str, int]) -> Dict[str, Any]:
    action_counts = Counter(state.action for state in states)
    reject_reasons = Counter(state.reject_reason for state in states if state.reject_reason)
    by_type: Dict[str, Dict[str, int]] = {}
    for query_type in sorted({state.query_type for state in states}):
        subset = [state for state in states if state.query_type == query_type]
        by_type[query_type] = {
            "keep": sum(1 for state in subset if state.action == "keep"),
            "regenerate": sum(1 for state in subset if state.action == "regenerate"),
            "reject": sum(1 for state in subset if state.action == "reject"),
        }

    return {
        "input_rows": len(states),
        "processed_movies": len({state.movie_id for state in states}),
        "action_counts": dict(action_counts),
        "by_query_type": by_type,
        "top_reject_reasons": dict(reject_reasons.most_common(10)),
        "duplicates": duplicate_stats,
    }


def run_cleaner(args: argparse.Namespace) -> Dict[str, Any]:
    raw_rows = _select_movie_rows(_read_jsonl(args.input), args.movies)
    movie_ids = {str(row.get("movie_id", "")) for row in raw_rows}
    logger.info("Loaded %d raw rows across %d movies", len(raw_rows), len(movie_ids))

    corpus = build_scene_corpus()
    corpus = {movie_id: entry for movie_id, entry in corpus.items() if movie_id in movie_ids}
    logger.info("Loaded %d source movies from the scene corpus", len(corpus))

    states: List[RowState] = []
    for row in raw_rows:
        movie_id = str(row.get("movie_id", ""))
        movie_title = str(row.get("movie_title", ""))
        query_type = str(row.get("query_type", ""))
        scene_idx_raw = row.get("scene_idx")
        scene_idx = int(scene_idx_raw) if isinstance(scene_idx_raw, int) or (isinstance(scene_idx_raw, str) and scene_idx_raw.isdigit()) else None
        original_query = "" if row.get("query") is None else str(row.get("query"))

        entry = corpus.get(movie_id)
        scene = _resolve_scene(entry, scene_idx) if query_type == "scene_summary" else None
        repair = repair_query(original_query)

        state = RowState(
            movie_id=movie_id,
            movie_title=movie_title,
            query_type=query_type,
            scene_idx=scene_idx,
            original_query=original_query,
            cleaned_query=repair.cleaned_query,
            repair_method=repair.repair_method,
            flags=list(repair.flags),
            scene=scene,
            overview=entry.overview if entry else None,
            source_missing=(entry is None or (query_type == "scene_summary" and scene is None)),
        )
        if repair.structural_reason:
            state.flags.append(repair.structural_reason)

        if state.cleaned_query:
            scored = score_query(
                state.cleaned_query,
                movie_title=movie_title,
                query_type=query_type,
                scene=scene,
            )
            state.cleaned_query = scored.query
            state.quality_score = scored.score
            state.flags = sorted(set(state.flags + scored.flags))
        else:
            state.quality_score = 0.0

        state.action = _initial_action(state, allow_regenerate=not args.no_regenerate)
        if state.action == "reject":
            if state.cleaned_query is None:
                state.reject_reason = repair.structural_reason or "missing_clean_query"
            else:
                state.reject_reason = "failed_quality_threshold"
        states.append(state)

    if not args.no_regenerate:
        qgen = QueryGenerator(api_key=args.api_key)

        for state in states:
            if state.action == "regenerate" and state.query_type == "scene_summary":
                _regenerate_scene_summary(state, qgen)

        synopsis_groups: Dict[str, List[RowState]] = defaultdict(list)
        for state in states:
            if state.action == "regenerate" and state.query_type == "synopsis":
                synopsis_groups[state.movie_id].append(state)
        for movie_id, group in synopsis_groups.items():
            entry = corpus.get(movie_id)
            if entry is None:
                for state in group:
                    _mark_rejected(state, "missing_movie_source")
                continue
            _regenerate_synopsis_group(group, entry, qgen, states)

    kept_rows, duplicate_stats = _deduplicate_kept_rows(states)

    cleaned_output = [_make_clean_output_row(state) for state in kept_rows]
    audit_output = [_make_audit_row(state) for state in states]
    rejected_output = [_make_rejected_row(state) for state in states if state.action == "reject"]
    report = _build_report(states, duplicate_stats)

    _write_jsonl(args.output, cleaned_output)
    _write_jsonl(args.audit, audit_output)
    _write_jsonl(args.rejected, rejected_output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "Cleaning complete: %d kept, %d regenerated, %d rejected",
        report["action_counts"].get("keep", 0),
        report["action_counts"].get("regenerate", 0),
        report["action_counts"].get("reject", 0),
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_cleaner(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
