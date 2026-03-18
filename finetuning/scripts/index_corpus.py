#!/usr/bin/env python3
"""CLI: Embed all scenes with a model and upsert into Qdrant.

Usage::

    python -m finetuning.scripts.index_corpus --model fede-embeddinggemma/round2
    python -m finetuning.scripts.index_corpus --model fede-embeddinggemma/round2 --movies 500
"""

from __future__ import annotations

import argparse
import logging

from finetuning.corpus.scene_corpus import build_scene_corpus
from finetuning.training.model import encode_documents, load_model

from vector_db.indexer import SceneRecord, ScriptIndexer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed scenes and index into Qdrant")
    parser.add_argument("--model", type=str, required=True, help="Model ID or checkpoint path")
    parser.add_argument("--movies", type=int, default=None, help="Max movies to index (default: all)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model = load_model(args.model)
    corpus = build_scene_corpus(max_movies=args.movies)
    indexer = ScriptIndexer()

    total_scenes = sum(len(e.scenes) for e in corpus.values())
    logger.info("Indexing %d movies, %d scenes …", len(corpus), total_scenes)

    indexed = 0
    for i, (movie_id, entry) in enumerate(corpus.items(), 1):
        scene_texts = [s.text for s in entry.scenes]
        embeddings = encode_documents(model, scene_texts, batch_size=args.batch_size)

        records = [
            SceneRecord(
                movie_id=scene.movie_id,
                movie_title=scene.movie_title,
                scene_id=scene.scene_id,
                scene_index=scene.scene_index,
                text=scene.text,
                embedding=emb.tolist(),
                scene_title=scene.scene_title,
                character_names=scene.character_names,
            )
            for scene, emb in zip(entry.scenes, embeddings)
        ]

        indexer.index_movie_batch(scenes=records, sentences=[])
        indexed += len(records)

        if i % 100 == 0:
            logger.info("  indexed %d / %d movies (%d scenes)", i, len(corpus), indexed)

    logger.info("Indexing complete: %d movies, %d scenes", len(corpus), indexed)


if __name__ == "__main__":
    main()
