"""Normalization helpers for graph ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from .graph_models import DroppedRelation, GraphEntity, MovieGraphDocument, NormalizedRelation
from .predicates import VALID_ENTITY_TYPES, VALID_PREDICATES

_NOISE_RE = re.compile(r"^[\W_]+$")


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return normalized or "unknown"


def stable_entity_id(movie_id: str, entity_type: str, canonical_name: str) -> str:
    return f"{movie_id}:{entity_type.lower()}:{slugify(canonical_name)}"


def stable_relation_id(
    movie_id: str,
    from_entity_id: str,
    predicate: str,
    to_entity_id: str,
    evidence: str,
) -> str:
    digest = hashlib.sha1(
        "||".join(
            [
                movie_id,
                from_entity_id,
                predicate,
                to_entity_id,
                normalize_whitespace(evidence),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return f"{movie_id}:rel:{digest[:16]}"


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _is_invalid_name(value: str) -> bool:
    cleaned = normalize_whitespace(value)
    return not cleaned or bool(_NOISE_RE.fullmatch(cleaned))


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def available_movie_ids(entities_dir: Path) -> list[str]:
    movie_ids = []
    for path in sorted(entities_dir.glob("*_entities.json")):
        movie_ids.append(path.name.replace("_entities.json", ""))
    return movie_ids


def load_movie_document(
    movie_id: str,
    entities_dir: Path,
    relations_dir: Path,
) -> MovieGraphDocument:
    entities_path = entities_dir / f"{movie_id}_entities.json"
    relations_path = relations_dir / f"{movie_id}_relations.json"
    if not entities_path.exists():
        raise FileNotFoundError(f"Missing entity file for movie_id={movie_id}")

    entity_payload = _read_json(entities_path)
    relation_payload = _read_json(relations_path) if relations_path.exists() else {"relations": []}
    source_file = entity_payload.get("file") or relation_payload.get("file") or f"{movie_id}.txt"
    title = Path(source_file).stem.replace("-", " ")

    entities = _normalize_entities(movie_id=movie_id, source_file=source_file, raw_entities=entity_payload.get("entities", []))
    entity_index = {entity.canonical_name: entity for entity in entities}
    relations, dropped = _normalize_relations(
        movie_id=movie_id,
        source_file=source_file,
        raw_relations=relation_payload.get("relations", []),
        entity_index=entity_index,
    )
    return MovieGraphDocument(
        movie_id=movie_id,
        title=title,
        source_file=source_file,
        entities=entities,
        relations=relations,
        dropped_relations=dropped,
    )


def _normalize_entities(
    *,
    movie_id: str,
    source_file: str,
    raw_entities: Iterable[dict],
) -> list[GraphEntity]:
    entities: list[GraphEntity] = []
    seen: set[str] = set()
    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue
        name = normalize_whitespace(str(raw.get("text", "")))
        entity_type = normalize_whitespace(str(raw.get("label", ""))).upper()
        if _is_invalid_name(name) or entity_type not in VALID_ENTITY_TYPES:
            continue
        entity_id = stable_entity_id(movie_id, entity_type, name)
        if entity_id in seen:
            continue
        seen.add(entity_id)
        entities.append(
            GraphEntity(
                entity_id=entity_id,
                movie_id=movie_id,
                canonical_name=name,
                entity_type=entity_type,
                source_file=source_file,
            )
        )
    entities.sort(key=lambda entity: (entity.entity_type, entity.canonical_name))
    return entities


def _normalize_relations(
    *,
    movie_id: str,
    source_file: str,
    raw_relations: Iterable[dict],
    entity_index: dict[str, GraphEntity],
) -> tuple[list[NormalizedRelation], list[DroppedRelation]]:
    relations: list[NormalizedRelation] = []
    dropped: list[DroppedRelation] = []
    seen_relation_ids: set[str] = set()
    for raw in raw_relations:
        reason = _relation_drop_reason(raw, entity_index)
        if reason is not None:
            dropped.append(DroppedRelation(movie_id=movie_id, reason=reason, raw_relation=raw if isinstance(raw, dict) else {"value": raw}))
            continue
        from_name = normalize_whitespace(raw["from"])
        to_name = normalize_whitespace(raw["to"])
        evidence = normalize_whitespace(raw["evidence"])
        predicate = raw["label"].strip().upper()
        from_entity = entity_index[from_name]
        to_entity = entity_index[to_name]
        relation_id = stable_relation_id(movie_id, from_entity.entity_id, predicate, to_entity.entity_id, evidence)
        if relation_id in seen_relation_ids:
            continue
        seen_relation_ids.add(relation_id)
        relations.append(
            NormalizedRelation(
                relation_id=relation_id,
                movie_id=movie_id,
                from_entity_id=from_entity.entity_id,
                to_entity_id=to_entity.entity_id,
                from_name=from_entity.canonical_name,
                from_type=from_entity.entity_type,
                to_name=to_entity.canonical_name,
                to_type=to_entity.entity_type,
                predicate=predicate,
                evidence=evidence,
                source_file=source_file,
            )
        )
    relations.sort(key=lambda relation: relation.relation_id)
    return relations, dropped


def _relation_drop_reason(raw: object, entity_index: dict[str, GraphEntity]) -> str | None:
    if not isinstance(raw, dict):
        return "malformed_relation"
    from_name = normalize_whitespace(str(raw.get("from", "")))
    to_name = normalize_whitespace(str(raw.get("to", "")))
    evidence = normalize_whitespace(str(raw.get("evidence", "")))
    predicate = normalize_whitespace(str(raw.get("label", ""))).upper()
    from_type = normalize_whitespace(str(raw.get("from_type", ""))).upper()
    to_type = normalize_whitespace(str(raw.get("to_type", ""))).upper()
    if _is_invalid_name(from_name) or _is_invalid_name(to_name):
        return "blank_entity_name"
    if not evidence:
        return "empty_evidence"
    if predicate not in VALID_PREDICATES:
        return "invalid_predicate"
    if from_type not in VALID_ENTITY_TYPES or to_type not in VALID_ENTITY_TYPES:
        return "invalid_entity_type"
    if from_name not in entity_index:
        return "missing_from_entity"
    if to_name not in entity_index:
        return "missing_to_entity"
    if from_name == to_name and predicate in {"BETRAYS", "TEACHES", "LIES_TO", "CONFRONTS", "KILLS"}:
        return "self_relation"
    return None
