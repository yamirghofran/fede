"""Build a per-movie scene corpus from tagged scripts and TMDB metadata.

Uses ``ScriptChunker`` from the ``preprocessing`` package (assumed merged
from ``feat/script-chunker``) to parse each tagged script into scene chunks.
Scenes shorter than ``MIN_SCENE_WORDS`` are dropped.  Each movie is paired
with its TMDB overview so downstream query generators can use it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from preprocessing.chunker import SceneChunk, ScriptChunker

from finetuning.config import (
    METADATA_PATH,
    MIN_SCENE_WORDS,
    TAGGED_SCRIPTS_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class MovieEntry:
    """A single movie with its scene chunks and TMDB overview."""

    movie_id: str
    movie_title: str
    overview: Optional[str]
    scenes: List[SceneChunk] = field(default_factory=list)


def _word_count(text: str) -> int:
    return len(text.split())


def load_metadata(metadata_path: Optional[Path] = None) -> dict:
    """Load ``clean_parsed_meta.json`` and return the raw dict."""
    path = metadata_path or METADATA_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_tagged_path(file_name: str, scripts_dir: Path) -> Optional[Path]:
    """Try to find the tagged script file for a movie.

    Looks for ``<file_name>_parsed.txt`` (the convention used by the
    preprocessing pipeline) and falls back to ``<file_name>.txt``.
    """
    for suffix in (f"{file_name}_parsed.txt", f"{file_name}.txt"):
        candidate = scripts_dir / suffix
        if candidate.exists():
            return candidate
    return None


def build_scene_corpus(
    max_movies: Optional[int] = None,
    metadata_path: Optional[Path] = None,
    scripts_dir: Optional[Path] = None,
    min_scene_words: int = MIN_SCENE_WORDS,
) -> Dict[str, MovieEntry]:
    """Parse all tagged scripts into a movie-keyed scene corpus.

    Args:
        max_movies: Cap on how many movies to load (``None`` = all).
        metadata_path: Override for ``clean_parsed_meta.json`` location.
        scripts_dir: Override for the tagged-scripts directory.
        min_scene_words: Drop scenes with fewer words than this.

    Returns:
        ``{movie_id: MovieEntry}`` mapping.  Movies whose tagged script
        cannot be found on disk are silently skipped.
    """
    meta = load_metadata(metadata_path)
    scripts_dir = scripts_dir or TAGGED_SCRIPTS_DIR
    corpus: Dict[str, MovieEntry] = {}
    loaded = 0

    for key, entry in meta.items():
        if max_movies is not None and loaded >= max_movies:
            break

        file_info = entry.get("file", {})
        tmdb_info = entry.get("tmdb", {})
        file_name = file_info.get("file_name")
        movie_title = file_info.get("name", key)

        if not file_name:
            continue

        tagged_path = _resolve_tagged_path(file_name, scripts_dir)
        if tagged_path is None:
            logger.debug("Tagged script not found for %s — skipping", file_name)
            continue

        try:
            chunker = ScriptChunker(movie_title, str(tagged_path))
            scene_chunks, _ = chunker.parse()
        except Exception:
            logger.warning("Failed to parse %s — skipping", tagged_path, exc_info=True)
            continue

        filtered = [s for s in scene_chunks if _word_count(s.text) >= min_scene_words]
        if not filtered:
            logger.debug("No scenes >= %d words for %s — skipping", min_scene_words, movie_title)
            continue

        movie_id = chunker.movie_id
        corpus[movie_id] = MovieEntry(
            movie_id=movie_id,
            movie_title=movie_title,
            overview=tmdb_info.get("overview"),
            scenes=filtered,
        )
        loaded += 1

    logger.info("Scene corpus built: %d movies, %d total scenes",
                len(corpus), sum(len(m.scenes) for m in corpus.values()))
    return corpus
