"""
Compute NDCG@5 from human-annotated grading CSV.

Expected input CSV columns:
    person_id, query_id, query_text, engine, rank, movie_key,
    movie_title, movie_overview, relevance_grade

relevance_grade must be filled with 0, 1, or 2.

Usage:
    python scripts/compute_ndcg.py --input evaluation/ndcg_study/grading_template.csv
    python scripts/compute_ndcg.py --input evaluation/ndcg_study/grading_template.csv \\
        --output evaluation/results/ndcg_results.csv --k 5
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

RESULTS_DIR = os.path.join(project_root, "evaluation", "results")


# NDCG helpers

def _dcg(grades: List[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(ranked_grades: List[float], k: int) -> float:
    dcg = _dcg(ranked_grades, k)
    ideal = _dcg(sorted(ranked_grades, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


# Inter-rater agreement (Krippendorff's alpha, ordinal)

def _krippendorff_alpha_ordinal(ratings_matrix: List[List[float]]) -> float:
    """
    Simplified Krippendorff's alpha for ordinal data.
    ratings_matrix: list of [rater1_grade, rater2_grade, ...] per item.
    Only items with >= 2 raters are included.
    """
    # Flatten to paired differences
    n_pairable = 0
    observed_disagreement = 0.0
    all_values = []

    for item_ratings in ratings_matrix:
        valid = [r for r in item_ratings if r is not None]
        all_values.extend(valid)
        if len(valid) < 2:
            continue
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                d = (valid[i] - valid[j]) ** 2
                observed_disagreement += d
                n_pairable += 1

    if n_pairable == 0 or not all_values:
        return float("nan")

    # Expected disagreement under null hypothesis
    n = len(all_values)
    expected_disagreement = 0.0
    for vi in all_values:
        for vj in all_values:
            expected_disagreement += (vi - vj) ** 2
    expected_disagreement /= n * (n - 1)

    if expected_disagreement == 0:
        return 1.0

    Do = observed_disagreement / n_pairable
    return 1.0 - (Do / expected_disagreement)


# Main logic

def _load_annotations(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grade_str = row.get("relevance_grade", "").strip()
            if grade_str == "":
                continue  # skip unfilled rows
            try:
                grade = float(grade_str)
            except ValueError:
                print(f"WARNING: Invalid grade '{grade_str}' for row {row}. Skipping.")
                continue
            if grade not in (0.0, 1.0, 2.0):
                print(f"WARNING: Grade {grade} not in {{0, 1, 2}} for row {row}. Skipping.")
                continue
            rows.append({
                "person_id": row["person_id"],
                "query_id": row["query_id"],
                "engine": row["engine"],
                "rank": int(row["rank"]),
                "movie_key": row["movie_key"],
                "grade": grade,
            })
    return rows


def _aggregate_grades(rows: list) -> Dict[Tuple[str, str, str], float]:
    """
    Average grades across raters for each (query_id, engine, movie_key).
    Returns {(query_id, engine, movie_key): avg_grade}
    """
    accumulator: Dict[Tuple, List[float]] = defaultdict(list)
    for r in rows:
        key = (r["query_id"], r["engine"], r["movie_key"])
        accumulator[key].append(r["grade"])
    return {k: sum(v) / len(v) for k, v in accumulator.items()}


def _build_ranked_results(rows: list, aggregated: dict) -> Dict[Tuple[str, str], List[Tuple[int, str, float]]]:
    """
    Build {(query_id, engine): [(rank, movie_key, avg_grade), ...]} sorted by rank.
    Rank is taken from the first rater's view (consistent since we use the same retrieval list).
    """
    # Use median rank across raters (they all see the same ranking for a given query+engine)
    rank_lookup: Dict[Tuple[str, str, str], int] = {}
    for r in rows:
        key = (r["query_id"], r["engine"], r["movie_key"])
        if key not in rank_lookup:
            rank_lookup[key] = r["rank"]

    result: Dict[Tuple[str, str], List[Tuple[int, str, float]]] = defaultdict(list)
    for (qid, engine, mkey), avg_grade in aggregated.items():
        rank = rank_lookup.get((qid, engine, mkey), 99)
        result[(qid, engine)].append((rank, mkey, avg_grade))

    # Sort by rank
    for key in result:
        result[key].sort(key=lambda x: x[0])

    return dict(result)


def _compute_ndcg_per_engine(
    ranked: Dict[Tuple[str, str], List],
    k: int,
) -> Dict[str, List[float]]:
    """Returns {engine: [ndcg_score_per_query]}"""
    engine_scores: Dict[str, List[float]] = defaultdict(list)
    for (qid, engine), items in ranked.items():
        grades = [grade for _, _, grade in items]
        score = ndcg_at_k(grades, k)
        engine_scores[engine].append(score)
    return dict(engine_scores)


def _compute_iaa(rows: list) -> Dict[str, float]:
    """Compute Krippendorff's alpha per engine."""
    # Group: {engine: {(qid, movie_key): {person_id: grade}}}
    grouped: Dict[str, Dict[Tuple, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        grouped[r["engine"]][(r["query_id"], r["movie_key"])][r["person_id"]] = r["grade"]

    iaa = {}
    for engine, item_ratings in grouped.items():
        matrix = [list(rater_grades.values()) for rater_grades in item_ratings.values()]
        iaa[engine] = _krippendorff_alpha_ordinal(matrix)
    return iaa


def main():
    parser = argparse.ArgumentParser(description="Compute NDCG@k from human grading CSV")
    parser.add_argument("--input", required=True, help="Path to filled grading_template.csv")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: evaluation/results/ndcg_results.csv",
    )
    parser.add_argument("--k", type=int, default=5, help="NDCG cutoff (default: 5)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(RESULTS_DIR, "ndcg_results.csv")

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("FEDE - NDCG COMPUTATION")
    print("=" * 60)
    print(f"Input : {args.input}")
    print(f"k     : {args.k}")

    rows = _load_annotations(args.input)
    if not rows:
        print("ERROR: No valid annotation rows found. Check that relevance_grade is filled.")
        sys.exit(1)

    print(f"Valid annotations: {len(rows)}")

    # Aggregate grades across raters
    aggregated = _aggregate_grades(rows)

    # Build ranked results
    ranked = _build_ranked_results(rows, aggregated)

    # NDCG per engine
    engine_scores = _compute_ndcg_per_engine(ranked, args.k)

    # Inter-annotator agreement
    iaa = _compute_iaa(rows)

    # Count raters per engine
    raters_per_engine: Dict[str, set] = defaultdict(set)
    queries_per_engine: Dict[str, set] = defaultdict(set)
    for r in rows:
        raters_per_engine[r["engine"]].add(r["person_id"])
        queries_per_engine[r["engine"]].add(r["query_id"])

    print("\n" + "-" * 35)
    print(f"{'Engine':<15} {'NDCG@' + str(args.k):>10}")
    print("-" * 35)

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    csv_rows = []

    for engine in sorted(engine_scores.keys()):
        scores = engine_scores[engine]
        mean_ndcg = sum(scores) / len(scores)
        variance = sum((s - mean_ndcg) ** 2 for s in scores) / len(scores)
        std_ndcg = math.sqrt(variance)
        alpha = iaa.get(engine, float("nan"))
        n_queries = len(queries_per_engine[engine])
        n_raters = len(raters_per_engine[engine])

        print(f"{engine:<15} {mean_ndcg*100:>9.1f}%")

        csv_rows.append({
            "engine": engine,
            "n_queries": n_queries,
            "n_raters": n_raters,
            f"ndcg@{args.k}": round(mean_ndcg, 4),
        })

    print("-" * 35)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "engine", "n_queries", "n_raters",
        f"ndcg@{args.k}",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
