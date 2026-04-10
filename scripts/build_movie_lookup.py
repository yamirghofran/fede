"""Build a movie key lookup table.

Usage:
    python scripts/build_movie_lookup.py
"""
import json
import re
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, "data", "scripts", "metadata", "clean_parsed_meta.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "evaluation", "evaluation_dataset", "movie_key_lookup.json")


def _simple_norm(s: str) -> str:
    """Remove all non-alphanumeric chars and lowercase."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _aliases_for(key: str, entry: dict) -> list[str]:
    """Generate candidate alias strings for a metadata entry."""
    candidates = set()

    # 1. The key itself
    candidates.add(key)

    # 2. The file display name (e.g. "10 Things I Hate About You")
    file_name = entry.get("file", {}).get("name", "")
    if file_name:
        candidates.add(_simple_norm(file_name))

    # 3. The TMDB title (may differ, e.g. "Twelve Monkeys" vs "12 Monkeys")
    tmdb_title = entry.get("tmdb", {}).get("title", "")
    if tmdb_title:
        candidates.add(_simple_norm(tmdb_title))

    # 4. Stripping trailing ", The" / ", A" / ", An" from titles then renormalizing
    for raw in (file_name, tmdb_title):
        cleaned = re.sub(r",?\s+(the|a|an)$", "", raw, flags=re.IGNORECASE).strip()
        if cleaned and cleaned.lower() != raw.lower():
            candidates.add(_simple_norm(cleaned))
        # Also try prepending "the" / "a" at the front
        for prefix in ("the", "a", "an"):
            prefixed = f"{prefix} {raw}".strip()
            candidates.add(_simple_norm(prefixed))

    # 5. The file_name field (hyphenated, e.g. "10-Things-I-Hate-About-You")
    script_file = entry.get("file", {}).get("file_name", "")
    if script_file:
        candidates.add(_simple_norm(script_file))

    return [c for c in candidates if c]


def build_lookup(metadata: dict) -> dict[str, str]:
    """Return {alias: canonical_key} with conflicts resolved to shortest key."""
    # alias -> list of (canonical_key, file_name) tuples
    alias_map: dict[str, list[tuple[str, str]]] = {}

    for key, entry in metadata.items():
        for alias in _aliases_for(key, entry):
            alias_map.setdefault(alias, []).append((key, entry.get("file", {}).get("name", key)))

    lookup: dict[str, str] = {}
    conflicts: list[tuple[str, list]] = []

    for alias, entries in alias_map.items():
        if len(entries) == 1:
            lookup[alias] = entries[0][0]
        else:
            # Conflict: prefer the exact match alias == key
            exact = [e for e in entries if e[0] == alias]
            if exact:
                lookup[alias] = exact[0][0]
            else:
                # Prefer shortest key (usually the cleaner one)
                entries.sort(key=lambda e: len(e[0]))
                lookup[alias] = entries[0][0]
                conflicts.append((alias, entries))

    return lookup, conflicts


def main():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Metadata: {len(metadata)} movies")

    lookup, conflicts = build_lookup(metadata)

    print(f"Lookup entries: {len(lookup)}")
    if conflicts:
        print(f"Conflicts resolved (showing first 10):")
        for alias, entries in conflicts[:10]:
            chosen = lookup[alias]
            others = [e[0] for e in entries if e[0] != chosen]
            print(f"  {alias!r} -> {chosen!r} (skipped: {others})")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, sort_keys=True, ensure_ascii=False)

    print(f"Saved to {OUTPUT_PATH}")

    # Verify the known-failing eval_queries movie IDs
    print("\nVerifying known-failing movie IDs from eval_queries.json:")
    failing = [
        "hitchhiker's_guide_to_the_galaxy,_the",
        "blood_and_wine",
        "fright_night_(1985)",
        "shape_of_water,_the",
        "flora_and_son",
        "haunting,_the",
        "30_minutes_or_less",
        "good_wife,_the___stripped",
        "united_states_vs._billie_holiday,_the",
        "boxtrolls,_the",
        "kids_are_alright,_the",
        "pacifier,_the",
        "last_of_the_mochican",
        "ladykillers,_the",
        "beauty_and_the_beast",
        "queen's_gambit,_the",
    ]
    for mid in failing:
        norm = _simple_norm(mid)
        found = lookup.get(norm)
        status = f"-> {found!r}" if found else "NOT FOUND"
        print(f"  {mid!r:45}  norm={norm!r:35}  {status}")


if __name__ == "__main__":
    main()
