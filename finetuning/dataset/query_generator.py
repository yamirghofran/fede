"""Synthetic query generation for the finetuning dataset.

Uses an OpenRouter-hosted LLM to produce three kinds of training queries:

* **Type A — synopsis queries**: given a TMDB overview, generate diverse
  natural-language search queries (plot / conflict / theme / relationship).
* **Type B — scene summaries**: given raw scene text, produce a 1-sentence
  plain-English summary suitable as a retrieval query.
* **Type C — paraphrases**: rewrite an existing query in alternative
  phrasings to cheaply multiply data volume.

All three share a single ``QueryGenerator`` class that owns the LLM client,
retry logic, rate-limit handling, checkpointing, and leakage detection.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from finetuning.config import (
    FINETUNING_DATA_DIR,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RATE_LIMIT_DELAY,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    QUERIES_PER_MOVIE_SYNOPSIS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYNOPSIS_PROMPT = """\
You are generating search queries for a movie retrieval system.

Given the following movie synopsis, generate {n} short user-style search queries.
Rules:
- Do not mention the movie title.
- Do not mention character names.
- Do not mention actor names.
- Do not use direct quotes.
- Keep each query to 1 sentence.
- Make them diverse: one plot-focused, one conflict-focused, one relationship-focused, one theme/mood-focused.

Return ONLY a JSON array of strings, no other text.

Synopsis:
{synopsis}"""

_SCENE_SUMMARY_PROMPT = """\
You are creating training data for a movie retrieval system.

Convert the following movie script scene into ONE realistic user search query.

The query should describe the most distinctive event, conflict, relationship, or revelation in the scene as a person might remember it when searching for a movie.

Rules:
- Exactly one sentence.
- Plain natural English.
- No screenplay formatting.
- No movie title.
- No character names.
- No actor names.
- No direct quotes.
- No copied phrases unless unavoidable.
- No invented details.
- Focus on the core narrative content, not surface visuals.
- Make it distinctive but natural.
- Avoid spoilers unless necessary to preserve the meaning of the scene.

If the scene is too generic to form a useful retrieval query, output only:
SKIP

Return only the final query or SKIP.

Scene:
{scene_text}"""

_PARAPHRASE_PROMPT = """\
Rewrite this search query in {n} different ways.
Rules:
- Keep the meaning the same.
- Do not add new facts.
- Keep each version to 1 sentence.

Return ONLY a JSON array of strings, no other text.

Query:
{query}"""


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "that", "this", "it", "its",
    "he", "she", "they", "we", "i", "you", "as", "if", "not", "no", "so",
    "up", "out", "about", "into", "than", "then", "when", "who", "which",
    "what", "s", "t",
})


def check_leakage(query: str, movie_title: str) -> bool:
    """Return ``True`` if the query leaks the movie title.

    A query is considered leaked if it contains any non-stopword token that
    appears in the title *and* the matching tokens cover more than half
    of the title's significant words.
    """
    title_words = [w for w in _normalize(movie_title).split() if w not in _STOPWORDS]
    if not title_words:
        return False
    query_norm = _normalize(query)
    matched = sum(1 for w in title_words if w in query_norm)
    return matched / len(title_words) > 0.5


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path() -> Path:
    return FINETUNING_DATA_DIR / "querygen_checkpoint.json"


def save_checkpoint(state: Dict) -> None:
    """Atomically write checkpoint state to disk."""
    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.rename(path)


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint if it exists, otherwise return ``None``."""
    path = _checkpoint_path()
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# QueryGenerator
# ---------------------------------------------------------------------------

class QueryGenerator:
    """OpenRouter LLM client for synthetic query generation.

    Owns retry logic, rate limiting, and leakage filtering.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = LLM_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
    ) -> None:
        resolved_key = api_key or os.getenv(OPENROUTER_API_KEY_ENV, "")
        if not resolved_key:
            raise ValueError(
                f"No API key provided.  Set the {OPENROUTER_API_KEY_ENV} "
                f"environment variable or pass api_key explicitly."
            )
        self._client = OpenAI(api_key=resolved_key, base_url=base_url)
        self._model = model

    # ----- low-level LLM call -----

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Send a single prompt with exponential-backoff retry."""
        for attempt in range(LLM_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                choice = resp.choices[0] if resp.choices else None
                if choice and choice.message and choice.message.content:
                    return choice.message.content.strip()
                logger.warning("Empty LLM response (attempt %d/%d)", attempt + 1, LLM_MAX_RETRIES)
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                wait = LLM_RATE_LIMIT_DELAY * (2 ** attempt)
                logger.warning("LLM error %s — retrying in %.1fs", exc, wait)
                time.sleep(wait)
            except Exception:
                logger.error("Unexpected LLM error (attempt %d/%d)", attempt + 1, LLM_MAX_RETRIES, exc_info=True)
                if attempt < LLM_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return None

    def _parse_json_array(self, raw: str) -> List[str]:
        """Extract a JSON string array from LLM output, tolerating markdown fences."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON output: %.120s", raw)
        return []

    # ----- public generation methods -----

    def generate_synopsis_queries(
        self,
        overview: str,
        movie_title: str,
        n: int = QUERIES_PER_MOVIE_SYNOPSIS,
    ) -> List[str]:
        """Type A: generate *n* diverse search queries from a TMDB overview.

        Queries that leak the movie title are silently discarded.
        """
        if not overview or not overview.strip():
            return []

        prompt = _SYNOPSIS_PROMPT.format(n=n, synopsis=overview)
        raw = self._call_llm(prompt)
        if raw is None:
            return []

        queries = self._parse_json_array(raw)
        return [q for q in queries if not check_leakage(q, movie_title)]

    def generate_scene_summary(
        self,
        scene_text: str,
        movie_title: str,
    ) -> Optional[str]:
        """Type B: convert a single scene into a realistic search query.

        Returns ``None`` if the LLM responds with ``SKIP``, generation
        fails, or the result leaks the movie title.
        """
        prompt = _SCENE_SUMMARY_PROMPT.format(scene_text=scene_text)
        raw = self._call_llm(prompt)
        if raw is None:
            return None
        summary = raw.strip().strip('"').strip("'")
        if summary.upper() == "SKIP":
            return None
        if check_leakage(summary, movie_title):
            return None
        return summary

    def generate_paraphrases(
        self,
        query: str,
        movie_title: str,
        n: int = 2,
    ) -> List[str]:
        """Type C: rephrase an existing query in *n* alternative wordings.

        Paraphrases that leak the movie title are silently discarded.
        """
        prompt = _PARAPHRASE_PROMPT.format(n=n, query=query)
        raw = self._call_llm(prompt)
        if raw is None:
            return []

        paraphrases = self._parse_json_array(raw)
        return [p for p in paraphrases if not check_leakage(p, movie_title)]

    def throttle(self) -> None:
        """Sleep for the configured rate-limit delay between LLM calls."""
        time.sleep(LLM_RATE_LIMIT_DELAY)
