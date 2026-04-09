"""Runtime settings for the FastAPI backend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[2]


class BackendSettings(BaseSettings):
    """Configuration for API startup, embedding, and search windows."""

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "FEDE API"
    app_version: str = "0.1.0"
    app_host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("API_HOST", "FEDE_API_HOST"))
    app_port: int = Field(default=8000, ge=1, le=65535, validation_alias=AliasChoices("API_PORT", "FEDE_API_PORT"))
    app_reload: bool = Field(default=False, validation_alias=AliasChoices("API_RELOAD", "FEDE_API_RELOAD"))

    embedding_model_id: str = Field(
        default="google/embeddinggemma-300m",
        validation_alias=AliasChoices("EMBEDDING_MODEL_ID", "FEDE_EMBEDDING_MODEL_ID"),
    )
    embedding_device: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_DEVICE", "FEDE_EMBEDDING_DEVICE"),
    )
    embedding_batch_size: int = Field(
        default=16,
        ge=1,
        le=512,
        validation_alias=AliasChoices("EMBEDDING_BATCH_SIZE", "FEDE_EMBEDDING_BATCH_SIZE"),
    )
    embedding_fp16: bool = Field(
        default=False,
        validation_alias=AliasChoices("EMBEDDING_FP16", "FEDE_EMBEDDING_FP16"),
    )

    default_top_k: int = Field(
        default=10,
        ge=1,
        le=200,
        validation_alias=AliasChoices("SEARCH_DEFAULT_TOP_K", "FEDE_SEARCH_DEFAULT_TOP_K"),
    )
    max_top_k: int = Field(
        default=25,
        ge=1,
        le=500,
        validation_alias=AliasChoices("SEARCH_MAX_TOP_K", "FEDE_SEARCH_MAX_TOP_K"),
    )
    default_sentence_pool: int = Field(
        default=100,
        ge=1,
        le=5000,
        validation_alias=AliasChoices(
            "SEARCH_DEFAULT_SENTENCE_POOL",
            "FEDE_SEARCH_DEFAULT_SENTENCE_POOL",
        ),
    )
    max_sentence_pool: int = Field(
        default=300,
        ge=1,
        le=10000,
        validation_alias=AliasChoices("SEARCH_MAX_SENTENCE_POOL", "FEDE_SEARCH_MAX_SENTENCE_POOL"),
    )
    movie_overfetch_factor: int = Field(
        default=3,
        ge=1,
        le=20,
        validation_alias=AliasChoices(
            "SEARCH_MOVIE_OVERFETCH_FACTOR",
            "FEDE_SEARCH_MOVIE_OVERFETCH_FACTOR",
        ),
    )
    graph_db_path: Path = Field(
        default=_ROOT / "data" / "grafeo" / "story_graph.db",
        validation_alias=AliasChoices("GRAPH_DB_PATH", "FEDE_GRAPH_DB_PATH"),
    )
    graph_entities_dir: Path = Field(
        default=_ROOT / "data" / "knowledge_graph" / "entities_clean",
        validation_alias=AliasChoices("GRAPH_ENTITIES_DIR", "FEDE_GRAPH_ENTITIES_DIR"),
    )
    graph_relations_dir: Path = Field(
        default=_ROOT / "data" / "knowledge_graph" / "relations",
        validation_alias=AliasChoices("GRAPH_RELATIONS_DIR", "FEDE_GRAPH_RELATIONS_DIR"),
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "FEDE_LLM_API_KEY"),
    )
    llm_api_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_URL", "FEDE_LLM_API_URL"),
    )
    llm_model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LLM_MODEL", "FEDE_LLM_MODEL"),
    )

    @field_validator("embedding_device", "llm_api_key", "llm_api_url", "llm_model", mode="before")
    @classmethod
    def _normalize_optional_str(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("graph_db_path", "graph_entities_dir", "graph_relations_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value):
        if value is None or isinstance(value, Path):
            return value
        return Path(str(value))
