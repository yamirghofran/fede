import argparse
import csv
import json
import math
import os
import random
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

QUERIES_PATH = os.path.join(project_root, "evaluation", "evaluation_dataset", "generated_queries.json")
METADATA_PATH = os.path.join(project_root, "data", "scripts", "metadata", "clean_parsed_meta.json")
OUTPUT_DIR = os.path.join(project_root, "evaluation", "ndcg_study")

TOP_K_PER_ENGINE = 5
SEED = 42


def _load_queries(path: str, n: int) -> list:
    """Load first n queries without lexical leakage."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_q = data.get("evaluation_queries", [])
    clean = [q for q in all_q if not q.get("validation", {}).get("has_leakage", False)]
    if len(clean) < n:
        print(f"WARNING: Only {len(clean)} clean queries available (need {n}). Using all.")
    return clean[:n]


def _load_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_overview(metadata: dict, movie_key: str) -> str:
    """Return TMDB overview for the movie, truncated to 200 chars."""
    entry = metadata.get(movie_key, {})
    overview = entry.get("tmdb", {}).get("overview", "")
    return overview[:200] if overview else ""


def _retrieve_results(queries: list, methods: list, api_url: str) -> dict:
    """
    Returns: {method: {query_id: [result_dicts]}}
    result_dict: {movie_key, movie_name, score, rank}
    """
    results = {m: {} for m in methods}

    for method in methods:
        print(f"\n[{method.upper()}] Retrieving top-{TOP_K_PER_ENGINE} for {len(queries)} queries...")
        if method == "bm25":
            from evaluation.baselines.bm25_baseline import BM25Retriever
            retriever = BM25Retriever()
        else:
            from evaluation.baselines.api_retriever import ApiRetriever
            retriever = ApiRetriever(mode=method, base_url=api_url)
            if not retriever.health_check():
                print(f"  WARNING: API not reachable at {api_url}. Results may be empty.")

        for i, q in enumerate(queries):
            qid = str(q["id"])
            retrieved = retriever.retrieve(q["query"], top_k=TOP_K_PER_ENGINE)
            results[method][qid] = retrieved
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(queries)} done")

    return results


def _build_assignment_matrix(queries: list, methods: list, n_people: int, queries_per_person: int) -> list:
    """
    Build assignment matrix: each person gets queries_per_person/n_methods queries per method.
    Each (query, method) pair is rated by exactly 2 people.

    Returns: List[{person_id, assignments: [{query_id, method}]}]
    """
    rng = random.Random(SEED)
    n_methods = len(methods)
    queries_per_method_per_person = queries_per_person // n_methods

    # Each (query, method) pair needs 2 raters
    # Total ratings per method: n_people * queries_per_method_per_person
    # Unique (query, method) pairs: total_ratings / 2
    ratings_per_method = n_people * queries_per_method_per_person
    unique_per_method = ratings_per_method // 2

    n_queries = len(queries)
    if unique_per_method > n_queries:
        print(f"WARNING: Need {unique_per_method} unique queries per method but only {n_queries} available.")
        unique_per_method = n_queries

    # Create pool of (query_id, method) pairs — each repeated twice for 2 raters
    pool = []
    for method in methods:
        selected_queries = queries[:unique_per_method]
        pairs = [(str(q["id"]), method) for q in selected_queries]
        pool.extend(pairs * 2)  # each pair rated by 2 people

    rng.shuffle(pool)

    # Assign to people: each person gets queries_per_person pairs
    # Constraint: no person sees the same query twice (regardless of method)
    assignments = []
    unassigned = list(pool)

    for person_idx in range(n_people):
        person_id = f"P{person_idx+1:02d}"
        person_assignments = []
        seen_queries = set()
        method_counts = {m: 0 for m in methods}
        remaining = []

        for pair in unassigned:
            qid, method = pair
            if (qid not in seen_queries
                    and method_counts[method] < queries_per_method_per_person
                    and len(person_assignments) < queries_per_person):
                person_assignments.append({"query_id": qid, "method": method})
                seen_queries.add(qid)
                method_counts[method] += 1
            else:
                remaining.append(pair)

        # Fill up if needed (fallback: allow method imbalance)
        for pair in list(remaining):
            if len(person_assignments) >= queries_per_person:
                break
            qid, method = pair
            if qid not in seen_queries:
                person_assignments.append({"query_id": qid, "method": method})
                seen_queries.add(qid)
                remaining.remove(pair)

        rng.shuffle(person_assignments)
        assignments.append({"person_id": person_id, "assignments": person_assignments})
        unassigned = remaining

    return assignments


def _build_grading_csv(
    assignments: list,
    queries: list,
    retrieval_results: dict,
    metadata: dict,
) -> list:
    """Build rows for grading_template.csv."""
    query_map = {str(q["id"]): q for q in queries}
    rows = []

    for person in assignments:
        pid = person["person_id"]
        for a in person["assignments"]:
            qid = a["query_id"]
            method = a["method"]
            q = query_map.get(qid)
            if not q:
                continue

            results = retrieval_results.get(method, {}).get(qid, [])
            for rank, r in enumerate(results, start=1):
                movie_key = r["movie_key"]
                overview = _get_overview(metadata, movie_key)
                rows.append({
                    "person_id": pid,
                    "query_id": qid,
                    "query_text": q["query"],
                    "engine": method,
                    "rank": rank,
                    "movie_key": movie_key,
                    "movie_title": r.get("movie_name", movie_key),
                    "movie_overview": overview,
                    "relevance_grade": "",  # to be filled by annotator (0, 1, or 2)
                })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate NDCG human annotation batches")
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=["bm25", "semantic", "hybrid"],
        help="Retrieval methods to include in the study",
    )
    parser.add_argument("--people", type=int, default=16, help="Number of annotators (default: 16)")
    parser.add_argument(
        "--queries-per-person",
        type=int,
        default=8,
        help="Queries per annotator (default: 8, must be divisible by number of methods)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="FEDE API base URL (for non-BM25 methods)",
    )
    args = parser.parse_args()

    n_methods = len(args.methods)
    if args.queries_per_person % n_methods != 0:
        print(f"ERROR: --queries-per-person ({args.queries_per_person}) must be divisible by "
              f"number of methods ({n_methods})")
        sys.exit(1)

    # How many unique queries needed
    queries_per_method_per_person = args.queries_per_person // n_methods
    unique_queries_needed = (args.people * queries_per_method_per_person) // 2
    unique_queries_needed = max(unique_queries_needed, 1)

    print("=" * 60)
    print("FEDE — NDCG BATCH GENERATION")
    print("=" * 60)
    print(f"Methods          : {', '.join(args.methods)}")
    print(f"People           : {args.people}")
    print(f"Queries/person   : {args.queries_per_person}")
    print(f"  Per method     : {queries_per_method_per_person}")
    print(f"Unique queries   : {unique_queries_needed} (2 raters each)")
    print(f"Top-k per engine : {TOP_K_PER_ENGINE}")
    total_rows = args.people * args.queries_per_person * TOP_K_PER_ENGINE
    print(f"CSV rows         : ~{total_rows} (excluding header)")

    print("\nLoading queries...")
    queries = _load_queries(QUERIES_PATH, unique_queries_needed)
    print(f"  Loaded {len(queries)} queries")

    print("Loading metadata...")
    metadata = _load_metadata(METADATA_PATH)

    # Retrieve results
    retrieval_results = _retrieve_results(queries, args.methods, args.api_url)

    # Build assignment matrix
    print("\nBuilding assignment matrix...")
    assignments = _build_assignment_matrix(
        queries, args.methods, args.people, args.queries_per_person
    )

    # Build CSV rows
    csv_rows = _build_grading_csv(assignments, queries, retrieval_results, metadata)

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    assignments_path = os.path.join(OUTPUT_DIR, "assignments.json")
    with open(assignments_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "methods": args.methods,
                    "n_people": args.people,
                    "queries_per_person": args.queries_per_person,
                    "top_k_per_engine": TOP_K_PER_ENGINE,
                    "seed": SEED,
                },
                "assignments": assignments,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nAssignments saved to: {assignments_path}")

    csv_path = os.path.join(OUTPUT_DIR, "grading_template.csv")
    fieldnames = [
        "person_id", "query_id", "query_text", "engine",
        "rank", "movie_key", "movie_title", "movie_overview", "relevance_grade",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Grading template  : {csv_path}  ({len(csv_rows)} rows)")

    print("\nInstructions for annotators:")
    print("  Fill 'relevance_grade' column:")
    print("    0 = Not relevant")
    print("    1 = Somewhat relevant")
    print("    2 = Highly relevant")
    print("\nWhen done, run:")
    print("  python scripts/compute_ndcg.py --input evaluation/ndcg_study/grading_template.csv")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
