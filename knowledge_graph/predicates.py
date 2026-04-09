"""Shared predicate definitions for relation extraction and graph ingestion."""

from __future__ import annotations

CHAR_CHAR_PREDICATES = {
    "BETRAYS",
    "TEACHES",
    "SAVES",
    "LIES_TO",
    "CONFRONTS",
    "ALLIES_WITH",
    "THREATENS",
    "RECONCILES_WITH",
    "KILLS",
    "AVENGES",
    "MANIPULATES",
    "PROTECTS",
    "SACRIFICES_FOR",
    "OWES",
    "LOVES",
    "HATES",
    "FORGIVES",
    "BLAMES",
    "ABANDONS",
    "ENVIES",
}

CHAR_CONCEPT_PREDICATES = {
    "WANTS",
    "LEARNS",
    "LOSES",
    "DISCOVERS",
    "BELIEVES",
    "DOUBTS",
    "FEARS",
    "REVEALS",
    "REJECTS",
    "MASTERS",
    "INHERITS",
    "SEEKS",
    "ACCEPTS",
    "POSSESSES",
    "DESTROYS",
    "CREATES",
}

TRANSFORMATION_PREDICATES = {
    "TRANSFORMS_INTO",
    "CORRUPTS",
    "REDEEMS",
    "REALIZES",
}

SOCIAL_PREDICATES = {
    "MARRIED_TO",
    "PARENT_OF",
    "CHILD_OF",
    "SIBLINGS_WITH",
    "WORKS_FOR",
    "LEADS",
}

VALID_PREDICATES = (
    CHAR_CHAR_PREDICATES
    | CHAR_CONCEPT_PREDICATES
    | TRANSFORMATION_PREDICATES
    | SOCIAL_PREDICATES
)

VALID_ENTITY_TYPES = {"PERSON", "EVENT", "ORG", "NORP", "GPE", "LOC", "FAC", "WORK_OF_ART"}
