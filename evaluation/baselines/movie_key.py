import json
import os
import re
from functools import lru_cache
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOOKUP_PATH = os.path.join(BASE_DIR, "evaluation", "evaluation_dataset", "movie_key_lookup.json")


def _simple_norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


@lru_cache(maxsize=1)
def _load_lookup() -> dict:
    if not os.path.exists(LOOKUP_PATH):
        raise FileNotFoundError(
            f"Movie key lookup not found at {LOOKUP_PATH}. "
            "Run: python scripts/build_movie_lookup.py"
        )
    with open(LOOKUP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_movie_key(movie_id: str) -> Optional[str]:
    """Return the canonical metadata key for *movie_id*, or None if unknown.

    Tries two passes:
    1. Simple normalization (strip all non-alphanumeric chars + lowercase).
    2. Lookup in the pre-built table which covers aliases, article variants,
       connector-word differences (e.g. "30minutesorless" -> "30minutesless").
    """
    norm = _simple_norm(movie_id)
    lookup = _load_lookup()
    # Direct hit on the normalized form
    if norm in lookup:
        return lookup[norm]
    # The normalized form might itself be a valid key (self-referential entries)
    return None
