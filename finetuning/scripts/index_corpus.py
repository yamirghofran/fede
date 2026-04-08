#!/usr/bin/env python3
"""CLI: Embed all scenes and sentences with a model and upsert into Qdrant.

Usage::

    python -m finetuning.scripts.index_corpus --model fede-embeddinggemma/round2
    python -m finetuning.scripts.index_corpus --model fede-embeddinggemma/round2 --movies 500
    python -m finetuning.scripts.index_corpus --model fede-embeddinggemma/round2 --scenes-only
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from preprocessing.chunker import SceneChunk, SentenceChunk, ScriptChunker

from finetuning.config import (
    METADATA_PATH,
    MIN_SCENE_WORDS,
    TAGGED_SCRIPTS_DIR,
)
from finetuning.training.model import encode_documents, load_model

from vector_db.indexer import SceneRecord, SentenceRecord, ScriptIndexer

logger = logging.getLogger(__name__)


def _resolve_tagged_path(file_name: str, scripts_dir: Path) -> Optional[Path]:
    for suffix in (f"{file_name}_parsed.txt", f"{file_name}.txt"):
        candidate = scripts_dir / suffix
        if candidate.exists():
            return candidate
    return None


def _parse_movie(
    movie_title: str,
    tagged_path: Path,
    min_scene_words: int,
) -> Optional[Tuple[str, List[SceneChunk], List[SentenceChunk]]]:
    """Parse a tagged script and return (movie_id, scenes, sentences)."""
    try:
        chunker = ScriptChunker(movie_title, str(tagged_path))
        scene_chunks, sentence_chunks = chunker.parse()
    except Exception:
        logger.warning("Failed to parse %s — skipping", tagged_path, exc_info=True)
        return None

    scenes = [s for s in scene_chunks if len(s.text.split()) >= min_scene_words]
    if not scenes:
        return None

    valid_scene_ids = {s.scene_id for s in scenes}
    sentences = [s for s in sentence_chunks if s.scene_id in valid_scene_ids]

    return chunker.movie_id, scenes, sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed scenes+sentences and index into Qdrant")
    parser.add_argument("--model", type=str, required=True, help="Model ID or checkpoint path")
    parser.add_argument("--movies", type=int, default=None, help="Max movies to index (default: all)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--scenes-only", action="store_true", help="Skip sentence indexing")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducibility")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model = load_model(args.model)
    indexer = ScriptIndexer()

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_items = list(meta.items())
    random.Random(args.seed).shuffle(meta_items)

    total_scenes = 0
    total_sentences = 0
    movies_indexed = 0

    for i, (key, entry) in enumerate(meta_items, 1):
        if args.movies is not None and movies_indexed >= args.movies:
            break

        file_info = entry.get("file", {})
        file_name = file_info.get("file_name")
        movie_title = file_info.get("name", key)
        if not file_name:
            continue

        tagged_path = _resolve_tagged_path(file_name, TAGGED_SCRIPTS_DIR)
        if tagged_path is None:
            continue

        parsed = _parse_movie(movie_title, tagged_path, MIN_SCENE_WORDS)
        if parsed is None:
            continue

        movie_id, scenes, sentences = parsed

        scene_texts = [s.text for s in scenes]
        scene_embs = encode_documents(model, scene_texts, batch_size=args.batch_size)

        scene_records = [
            SceneRecord(
                movie_id=sc.movie_id,
                movie_title=sc.movie_title,
                scene_id=sc.scene_id,
                scene_index=sc.scene_index,
                text=sc.text,
                embedding=emb.tolist(),
                scene_title=sc.scene_title,
                character_names=sc.character_names,
            )
            for sc, emb in zip(scenes, scene_embs)
        ]

        sentence_records: List[SentenceRecord] = []
        if not args.scenes_only and sentences:
            sent_texts = [s.text for s in sentences]
            sent_embs = encode_documents(model, sent_texts, batch_size=args.batch_size)

            sentence_records = [
                SentenceRecord(
                    movie_id=s.movie_id,
                    movie_title=s.movie_title,
                    scene_id=s.scene_id,
                    scene_index=s.scene_index,
                    text=s.text,
                    line_type=s.line_type,
                    position_in_script=s.position_in_script,
                    embedding=emb.tolist(),
                    character_name=s.character_name,
                )
                for s, emb in zip(sentences, sent_embs)
            ]

        indexer.index_movie_batch(scenes=scene_records, sentences=sentence_records)
        total_scenes += len(scene_records)
        total_sentences += len(sentence_records)
        movies_indexed += 1

        if movies_indexed % 50 == 0:
            logger.info(
                "  %d movies indexed (%d scenes, %d sentences)",
                movies_indexed, total_scenes, total_sentences,
            )

    logger.info(
        "Indexing complete: %d movies, %d scenes, %d sentences",
        movies_indexed, total_scenes, total_sentences,
    )


if __name__ == "__main__":
    main()
