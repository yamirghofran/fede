from __future__ import annotations

import time
from typing import Dict, List, Literal

import requests

MODE = Literal["semantic", "hybrid"]


class ApiRetriever:
    """
    Calls the FEDE API and normalizes results to:
        [{movie_key, movie_name, score, snippet}, ...]

    Modes:
        semantic → POST /search
        hybrid   → POST /query  {use_semantic=True, use_graph=True}
    """

    def __init__(self, mode: MODE, base_url: str = "http://localhost:8000", timeout: int = 120):
        if mode not in ("semantic", "hybrid"):
            raise ValueError(f"Unknown mode '{mode}'. Choose: semantic, hybrid")
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Returns list of dicts with keys: movie_key, movie_name, score, snippet.
        Returns [] on error (logs to stderr).
        """
        try:
            if self.mode == "semantic":
                return self._retrieve_semantic(query, top_k)
            else:  # hybrid
                return self._retrieve_query(query, top_k)
        except requests.exceptions.ConnectionError:
            print(f"[ApiRetriever] ERROR: Cannot connect to {self.base_url}. Is the API running?")
            return []
        except Exception as e:
            print(f"[ApiRetriever] ERROR for query '{query[:60]}': {e}")
            return []

    def _retrieve_semantic(self, query: str, top_k: int) -> List[Dict]:
        resp = requests.post(
            f"{self.base_url}/search",
            json={"query": query, "top_k": top_k},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", []):
            snippet = ""
            if r.get("best_scene"):
                snippet = r["best_scene"].get("text", "")[:150]
            results.append({
                "movie_key": r["movie_id"],
                "movie_name": r["movie_title"],
                "score": r["score"],
                "snippet": snippet,
            })
        return results

    def _retrieve_query(self, query: str, top_k: int) -> List[Dict]:
        resp = requests.post(
            f"{self.base_url}/query",
            json={
                "query": query,
                "top_k": top_k,
                "use_semantic": True,
                "use_graph": True,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", []):
            # Prefer best_scene snippet, fall back to graph evidence
            snippet = ""
            if r.get("best_scene"):
                snippet = r["best_scene"].get("text", "")[:150]
            elif r.get("graph_matches"):
                evs = r["graph_matches"][0].get("evidences", [])
                snippet = evs[0][:150] if evs else ""
            results.append({
                "movie_key": r["movie_id"],
                "movie_name": r["movie_title"],
                "score": r["score"],
                "snippet": snippet,
            })
        return results

    def health_check(self) -> bool:
        """Returns True if API is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/healthz", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
