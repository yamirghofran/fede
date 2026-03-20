"""Synthetic query generation for the finetuning dataset.

Uses an OpenRouter-hosted LLM to produce three kinds of training queries:

* **Type A — synopsis queries**: given a TMDB overview, generate diverse
  natural-language search queries (plot / conflict / theme / relationship).
* **Type B — scene summaries**: given raw scene text, produce a 1-sentence
  plain-English summary suitable as a retrieval query.
* **Type C — paraphrases**: rewrite an existing query in alternative
  phrasings to cheaply multiply data volume.

Both a synchronous ``QueryGenerator`` and an asynchronous
``AsyncQueryGenerator`` are provided.  The async version uses a global
``_AsyncRateLimiter`` to cap calls/second while allowing many coroutines
to run concurrently — this gives a 4–8× wall-clock speedup over the
sequential sync version for large datasets.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

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
You generate training data for a semantic search engine over movie scenes.

Task: Read the synopsis and write {n} distinct short queries a real user might type when searching for scenes from this kind of film.

Requirements for each query:
- One sentence, natural web-search style (not a critic's logline or marketing blurb).
- Under ~25 words.
- Spread angles across the set: e.g. plot situation, central conflict, relationship dynamic, theme or tone — avoid near-duplicates.
- Use only generic wording: do NOT use the film title, character names, actor names, or other proper nouns taken from the synopsis.
- No dialogue, no quotation marks, no "movie about" meta-phrasing unless unavoidable.

Output must be valid JSON only (no markdown fences, no commentary). Use exactly this shape with exactly {n} strings:
{{"queries": ["...", "..."]}}

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
You generate training data for a semantic search engine.

Task: Rewrite the search query below in {n} different ways. Same search intent and specificity; different surface wording.

Rules:
- Preserve meaning: do not add or remove facts, entities, or plot points.
- One sentence per rewrite.
- Vary structure: mix short vs slightly longer phrasings, statements vs questions, and synonyms — but stay faithful to the original.

Output must be valid JSON only (no markdown fences, no commentary). Use exactly this shape with exactly {n} strings:
{{"paraphrases": ["...", "..."]}}

Original query:
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
    """Return ``True`` if the query leaks the movie title."""
    title_words = [w for w in _normalize(movie_title).split() if w not in _STOPWORDS]
    if not title_words:
        return False
    query_norm = _normalize(query)
    matched = sum(1 for w in title_words if w in query_norm)
    return matched / len(title_words) > 0.5


# ---------------------------------------------------------------------------
# Shared JSON parsing (module-level so both sync and async classes use it)
# ---------------------------------------------------------------------------

def _parse_json_string_list(raw: str, *, preferred_key: Optional[str] = None) -> List[str]:
    """Parse a JSON list or object like ``{"queries": [...]}`` from LLM output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, dict):
            keys_to_try: List[str] = []
            if preferred_key:
                keys_to_try.append(preferred_key)
            keys_to_try.extend(
                k for k in ("queries", "paraphrases", "items", "results") if k not in keys_to_try
            )
            for key in keys_to_try:
                if key in parsed and isinstance(parsed[key], list):
                    return [str(item).strip() for item in parsed[key] if str(item).strip()]
            for value in parsed.values():
                if isinstance(value, list) and value:
                    return [str(item).strip() for item in value if str(item).strip()]
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON output: %.120s", raw)
    return []


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
# Async rate limiter
# ---------------------------------------------------------------------------

class _AsyncRateLimiter:
    """Ensures a minimum wall-clock interval between successive LLM calls.

    A single instance is shared across all concurrent coroutines so that
    the global call rate never exceeds ``1 / min_interval`` calls/second,
    regardless of concurrency level.
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_call_time: float = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait_time = self._min_interval - (now - self._last_call_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call_time = asyncio.get_event_loop().time()


# ---------------------------------------------------------------------------
# Synchronous QueryGenerator (kept for backward compatibility)
# ---------------------------------------------------------------------------

class QueryGenerator:
    """Synchronous OpenRouter LLM client for synthetic query generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = LLM_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
    ) -> None:
        resolved_key = api_key or os.getenv(OPENROUTER_API_KEY_ENV, "")
        if not resolved_key:
            raise ValueError(
                f"No API key provided. Set {OPENROUTER_API_KEY_ENV} or pass api_key."
            )
        self._client = OpenAI(api_key=resolved_key, base_url=base_url)
        self._model = model

    def _call_llm(
        self,
        prompt: str,
        *,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        for attempt in range(LLM_MAX_RETRIES):
            try:
                text = self._call_llm_once(prompt, response_format=response_format)
                if text:
                    return text
                if response_format is not None:
                    text = self._call_llm_once(prompt, response_format=None)
                    if text:
                        return text
                logger.warning("Empty LLM response (attempt %d/%d)", attempt + 1, LLM_MAX_RETRIES)
            except BadRequestError as exc:
                if response_format is not None:
                    logger.warning("JSON mode rejected (%s); retrying without", exc)
                    try:
                        text = self._call_llm_once(prompt, response_format=None)
                        if text:
                            return text
                    except BadRequestError:
                        pass
                if attempt < LLM_MAX_RETRIES - 1:
                    time.sleep(LLM_RATE_LIMIT_DELAY * (2 ** attempt))
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                wait = LLM_RATE_LIMIT_DELAY * (2 ** attempt)
                logger.warning("LLM error %s — retrying in %.1fs", exc, wait)
                time.sleep(wait)
            except Exception:
                logger.error("Unexpected LLM error (attempt %d/%d)", attempt + 1, LLM_MAX_RETRIES, exc_info=True)
                if attempt < LLM_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return None

    def _call_llm_once(
        self,
        prompt: str,
        *,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0] if resp.choices else None
        if choice and choice.message and choice.message.content:
            return choice.message.content.strip()
        return None

    def generate_synopsis_queries(
        self,
        overview: str,
        movie_title: str,
        n: int = QUERIES_PER_MOVIE_SYNOPSIS,
    ) -> List[str]:
        if not overview or not overview.strip():
            return []
        prompt = _SYNOPSIS_PROMPT.format(n=n, synopsis=overview)
        raw = self._call_llm(prompt, response_format={"type": "json_object"})
        if raw is None:
            return []
        return [q for q in _parse_json_string_list(raw, preferred_key="queries")
                if not check_leakage(q, movie_title)]

    def generate_scene_summary(self, scene_text: str, movie_title: str) -> Optional[str]:
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

    def generate_paraphrases(self, query: str, movie_title: str, n: int = 2) -> List[str]:
        prompt = _PARAPHRASE_PROMPT.format(n=n, query=query)
        raw = self._call_llm(prompt, response_format={"type": "json_object"})
        if raw is None:
            return []
        return [p for p in _parse_json_string_list(raw, preferred_key="paraphrases")
                if not check_leakage(p, movie_title)]

    def throttle(self) -> None:
        time.sleep(LLM_RATE_LIMIT_DELAY)


# ---------------------------------------------------------------------------
# Async QueryGenerator
# ---------------------------------------------------------------------------

class AsyncQueryGenerator:
    """Async OpenRouter LLM client for concurrent synthetic query generation.

    Uses a single ``_AsyncRateLimiter`` shared across all coroutines to cap
    the global call rate at ``1 / min_interval`` calls/second while allowing
    many movies to be processed in parallel.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = LLM_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        min_interval: float = LLM_RATE_LIMIT_DELAY,
    ) -> None:
        resolved_key = api_key or os.getenv(OPENROUTER_API_KEY_ENV, "")
        if not resolved_key:
            raise ValueError(
                f"No API key provided. Set {OPENROUTER_API_KEY_ENV} or pass api_key."
            )
        self._client = AsyncOpenAI(api_key=resolved_key, base_url=base_url)
        self._model = model
        self._rate_limiter = _AsyncRateLimiter(min_interval)

    async def _call_llm_once(
        self,
        prompt: str,
        *,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0] if resp.choices else None
        if choice and choice.message and choice.message.content:
            return choice.message.content.strip()
        return None

    async def _call_llm(
        self,
        prompt: str,
        *,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        for attempt in range(LLM_MAX_RETRIES):
            await self._rate_limiter.wait()
            try:
                text = await self._call_llm_once(prompt, response_format=response_format)
                if text:
                    return text
                if response_format is not None:
                    text = await self._call_llm_once(prompt, response_format=None)
                    if text:
                        logger.warning("Fell back to non-JSON mode after empty structured response")
                        return text
                logger.warning("Empty LLM response (attempt %d/%d)", attempt + 1, LLM_MAX_RETRIES)
            except BadRequestError as exc:
                if response_format is not None:
                    logger.warning("JSON mode rejected (%s); retrying without", exc)
                    try:
                        text = await self._call_llm_once(prompt, response_format=None)
                        if text:
                            return text
                    except BadRequestError:
                        pass
                if attempt < LLM_MAX_RETRIES - 1:
                    await asyncio.sleep(LLM_RATE_LIMIT_DELAY * (2 ** attempt))
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                wait = LLM_RATE_LIMIT_DELAY * (2 ** attempt)
                logger.warning("LLM error %s — retrying in %.1fs", exc, wait)
                await asyncio.sleep(wait)
            except Exception:
                logger.error("Unexpected LLM error (attempt %d/%d)", attempt + 1, LLM_MAX_RETRIES, exc_info=True)
                if attempt < LLM_MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        return None

    async def generate_synopsis_queries(
        self,
        overview: str,
        movie_title: str,
        n: int = QUERIES_PER_MOVIE_SYNOPSIS,
    ) -> List[str]:
        if not overview or not overview.strip():
            return []
        prompt = _SYNOPSIS_PROMPT.format(n=n, synopsis=overview)
        raw = await self._call_llm(prompt, response_format={"type": "json_object"})
        if raw is None:
            return []
        return [q for q in _parse_json_string_list(raw, preferred_key="queries")
                if not check_leakage(q, movie_title)]

    async def generate_scene_summary(self, scene_text: str, movie_title: str) -> Optional[str]:
        prompt = _SCENE_SUMMARY_PROMPT.format(scene_text=scene_text)
        raw = await self._call_llm(prompt)
        if raw is None:
            return None
        summary = raw.strip().strip('"').strip("'")
        if summary.upper() == "SKIP":
            return None
        if check_leakage(summary, movie_title):
            return None
        return summary

    async def generate_paraphrases(self, query: str, movie_title: str, n: int = 2) -> List[str]:
        prompt = _PARAPHRASE_PROMPT.format(n=n, query=query)
        raw = await self._call_llm(prompt, response_format={"type": "json_object"})
        if raw is None:
            return []
        return [p for p in _parse_json_string_list(raw, preferred_key="paraphrases")
                if not check_leakage(p, movie_title)]
