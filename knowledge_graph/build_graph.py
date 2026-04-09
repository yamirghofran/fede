"""CLI entrypoint for building the Grafeo knowledge graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from vector_db.api.settings import BackendSettings

from .graph_store import KnowledgeGraphService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Grafeo knowledge graph from extracted entities and relations.")
    parser.add_argument("--movie-id", help="Reload a single movie by id", default=None)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the full graph from scratch")
    args = parser.parse_args()

    settings = BackendSettings()
    service = KnowledgeGraphService(
        db_path=settings.graph_db_path,
        entities_dir=settings.graph_entities_dir,
        relations_dir=settings.graph_relations_dir,
    )
    summary = service.build(movie_id=args.movie_id, rebuild=args.rebuild)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
