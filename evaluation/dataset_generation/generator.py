from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, RateLimitError, APIConnectionError

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from evaluation.dataset_generation.checkpoint_manager import CheckpointManager
from evaluation.dataset_generation.config import (
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PATH,
    DEFAULT_MODEL,
    EVAL_QUERIES_PATH,
    EXTENDED_IDX_BASE,
    GENERATION_PROMPT,
    LLM_API_BASE,
    MAX_RETRIES,
    METADATA_PATH,
    MIN_RELATIONS,
    QUERIES_PER_MOVIE,
    RATE_LIMIT_DELAY,
    RELATIONS_DIR,
    VALIDATION_STRICTNESS,
)
from evaluation.dataset_generation.validator import QueryValidator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# Dataset I/O

def load_dataset(path: str) -> List[Dict]:
    """Load eval dataset; supports both flat-list and nested formats."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("evaluation_queries", [])


def save_dataset(queries: List[Dict], path: str) -> None:
    """Save flat-list format (eval_queries.json style)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)


# Relation helpers

def _relations_file_for_movie(movie_title: str) -> Optional[str]:
    """Find the relations file for a movie by matching title to filename."""
    if not os.path.isdir(RELATIONS_DIR):
        return None
    title_norm = re.sub(r"[^a-z0-9]", "", movie_title.lower())
    for fname in os.listdir(RELATIONS_DIR):
        if not fname.endswith("_relations.json"):
            continue
        base = fname.replace("_relations.json", "")
        base_norm = re.sub(r"[^a-z0-9]", "", base.lower())
        if base_norm == title_norm:
            return os.path.join(RELATIONS_DIR, fname)
    return None


def load_relations(movie_title: str) -> List[Dict]:
    """Return the list of narrative relations for a movie, or [] if not found."""
    path = _relations_file_for_movie(movie_title)
    if not path:
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return data.get("relations", [])


def _format_relations(relations: List[Dict], max_relations: int = 20) -> str:
    """Format relations as a readable text block for the LLM prompt."""
    lines = []
    for r in relations[:max_relations]:
        label = r.get("label", "?")
        from_ = r.get("from", "?")
        to = r.get("to", "?")
        evidence = r.get("evidence", "").strip().replace("\n", " ")[:80]
        lines.append(f'  {from_} --{label}--> {to}   ("{evidence}")')
    return "\n".join(lines)


# Query coverage helpers

def _movie_title_from_query(q: Dict) -> str:
    return q.get("movie_title") or q.get("movie_name", "")


def _already_covered(queries: List[Dict], movie_title: str) -> bool:
    """Return True if dataset already contains extended queries for this movie."""
    title_norm = movie_title.lower().strip()
    return any(
        _movie_title_from_query(q).lower().strip() == title_norm
        and q.get("_scene_idx", 0) >= EXTENDED_IDX_BASE
        for q in queries
    )


def _next_scene_idx(queries: List[Dict], movie_title: str) -> int:
    """Return the next available _scene_idx for extended queries of a movie."""
    title_norm = movie_title.lower().strip()
    existing = [
        q.get("_scene_idx", 0)
        for q in queries
        if _movie_title_from_query(q).lower().strip() == title_norm
        and q.get("_scene_idx", 0) >= EXTENDED_IDX_BASE
    ]
    return max(existing, default=EXTENDED_IDX_BASE - 1) + 1


def _movie_id_from_title(movie_title: str, metadata: Dict) -> str:
    """Best-effort: find the metadata key for a movie title."""
    title_norm = re.sub(r"[^a-z0-9]", "", movie_title.lower())
    for key, entry in metadata.items():
        name = entry.get("file", {}).get("name", "")
        if re.sub(r"[^a-z0-9]", "", name.lower()) == title_norm:
            return key
    return re.sub(r"[^a-z0-9_]", "_", movie_title.lower()).strip("_")


# Query generation

class QueryGenerator:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set LLM_API_KEY or OPENAI_API_KEY in .env")
        self._client = OpenAI(api_key=api_key, base_url=LLM_API_BASE)
        self._model = model

    def generate(self, movie_title: str, relations: List[Dict], n: int = QUERIES_PER_MOVIE) -> List[str]:
        """Generate n queries for a movie using its narrative relations."""
        relations_text = _format_relations(relations)
        prompt = GENERATION_PROMPT.format(
            movie_title=movie_title,
            relations_text=relations_text,
            n=n,
        )
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    timeout=60,
                )
                raw = resp.choices[0].message.content.strip()
                queries = [line.strip() for line in raw.splitlines() if line.strip()]
                return queries[:n]
            except RateLimitError:
                wait = 30 * attempt
                print(f"  Rate limit - waiting {wait}s")
                time.sleep(wait)
            except (APITimeoutError, APIConnectionError) as e:
                if attempt == MAX_RETRIES:
                    logger.warning("LLM call failed after %d retries: %s", MAX_RETRIES, e)
                    return []
                time.sleep(5 * attempt)
        return []


# Main pipeline

def run(
    target_movies: Optional[List[str]] = None,
    queries_per_movie: int = QUERIES_PER_MOVIE,
    validate_only: bool = False,
    resume: bool = False,
    output_path: str = EVAL_QUERIES_PATH,
) -> None:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    queries = load_dataset(output_path)
    print(f"Loaded {len(queries)} existing queries from {output_path}")

    if validate_only:
        _run_validation(queries, metadata)
        return

    all_movies = _discover_movies()
    print(f"Movies with relation data: {len(all_movies)}")

    if target_movies:
        requested_norm = {re.sub(r"[^a-z0-9]", "", m.lower()) for m in target_movies}
        candidates = {
            title: path for title, path in all_movies.items()
            if re.sub(r"[^a-z0-9]", "", title.lower()) in requested_norm
        }
        not_found = requested_norm - {re.sub(r"[^a-z0-9]", "", t.lower()) for t in candidates}
        if not_found:
            print(f"WARNING: no relation data found for: {not_found}")
    else:
        candidates = all_movies

    candidates = {
        title: path for title, path in candidates.items()
        if _count_relations(path) >= MIN_RELATIONS
    }

    if not target_movies:
        candidates = {
            title: path for title, path in candidates.items()
            if not _already_covered(queries, title)
        }

    print(f"Movies to process: {len(candidates)}")
    if not candidates:
        print("Nothing to do - all candidate movies already have extended queries.")
        _run_validation(queries, metadata)
        return

    checkpoint_mgr = CheckpointManager(CHECKPOINT_PATH, CHECKPOINT_INTERVAL)
    new_queries: List[Dict] = []

    if resume and checkpoint_mgr.has_checkpoint():
        cp = checkpoint_mgr.load_checkpoint()
        if cp:
            new_queries = cp.get("queries", [])
            done_titles = {q.get("movie_title", "") for q in new_queries}
            candidates = {t: p for t, p in candidates.items() if t not in done_titles}
            print(f"Resuming - {len(new_queries)} queries loaded from checkpoint")

    generator = QueryGenerator()
    validator = QueryValidator(metadata, strictness=VALIDATION_STRICTNESS)

    total = len(candidates)
    for i, (movie_title, _) in enumerate(candidates.items(), 1):
        relations = load_relations(movie_title)
        print(f"[{i}/{total}] {movie_title} ({len(relations)} relations)")

        raw_texts = generator.generate(movie_title, relations, n=queries_per_movie)
        if not raw_texts:
            print(f"  [!] LLM returned nothing")
            time.sleep(RATE_LIMIT_DELAY)
            continue

        movie_id = _movie_id_from_title(movie_title, metadata)
        start_idx = _next_scene_idx(queries + new_queries, movie_title)

        for j, text in enumerate(raw_texts):
            result = validator.check_lexical_leakage(text, movie_title)
            status = "[flagged]" if result["has_leakage"] else "[ok]"
            print(f"  {status}  {text[:80]}")
            new_queries.append({
                "query": text,
                "movie_id": movie_id,
                "movie_title": movie_title,
                "_scene_idx": start_idx + j,
                "validation": result,
            })

        time.sleep(RATE_LIMIT_DELAY)

        if checkpoint_mgr.should_checkpoint(i):
            checkpoint_mgr.save_checkpoint(new_queries, i, total)

    if new_queries:
        queries.extend(new_queries)
        save_dataset(queries, output_path)
        print(f"\nAdded {len(new_queries)} queries -> {len(queries)} total")
        checkpoint_mgr.clear_checkpoint()
    else:
        print("\nNo new queries generated.")

    _run_validation(queries, metadata)


def _discover_movies() -> Dict[str, str]:
    """Return {movie_title: relations_file_path} for all available relation files."""
    result = {}
    if not os.path.isdir(RELATIONS_DIR):
        return result
    for fname in sorted(os.listdir(RELATIONS_DIR)):
        if not fname.endswith("_relations.json"):
            continue
        title = fname.replace("_relations.json", "").replace("-", " ")
        result[title] = os.path.join(RELATIONS_DIR, fname)
    return result


def _count_relations(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return len(json.load(f).get("relations", []))
    except Exception:
        return 0


def _run_validation(queries: List[Dict], metadata: Dict) -> None:
    validator = QueryValidator(metadata, strictness=VALIDATION_STRICTNESS)
    report = validator.validate_batch(queries)
    print(f"\nValidation report:")
    print(f"  Total  : {report['total']}")
    print(f"  Passed : {report['passed']}")
    print(f"  Flagged: {report['flagged']}  (leakage score >= threshold)")
    if report["flagged_queries"]:
        print("  Flagged queries (first 5):")
        for q in report["flagged_queries"][:5]:
            print(f"    [{q.get('id', '?')}] {q['query'][:70]}  score={q['leakage_score']:.2f}")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate queries and append to eval_queries.json"
    )
    parser.add_argument(
        "--movies",
        nargs="+",
        metavar="TITLE",
        help="Movie titles to generate for (default: all with sufficient relation data)",
    )
    parser.add_argument(
        "--queries-per-movie",
        type=int,
        default=QUERIES_PER_MOVIE,
        help=f"Queries to generate per movie (default: {QUERIES_PER_MOVIE})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip generation and only run validation on existing queries",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--output",
        default=EVAL_QUERIES_PATH,
        help=f"Path to eval_queries.json (default: {EVAL_QUERIES_PATH})",
    )
    args = parser.parse_args()

    run(
        target_movies=args.movies,
        queries_per_movie=args.queries_per_movie,
        validate_only=args.validate_only,
        resume=args.resume,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
