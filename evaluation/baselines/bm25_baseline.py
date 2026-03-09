import json
import logging
import os
import re
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_PATH = os.path.join(BASE_DIR, "data", "scripts", "metadata", "clean_parsed_meta.json")
SCRIPTS_BASE_PATH = os.path.join(BASE_DIR, "data", "scripts", "filtered")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "it", "its", "he", "she", "they", "we", "i", "you", "his", "her",
    "their", "our", "my", "your", "as", "if", "not", "no", "so", "up",
    "out", "about", "into", "than", "then", "when", "who", "which", "what",
    "s", "t",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _read_script(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def _build_corpus(metadata: Dict, scripts_base: str) -> Tuple[List[str], List[str], List[List[str]]]:
    movie_keys, movie_names, tokenized = [], [], []
    missing = 0

    for key, entry in metadata.items():
        file_info = entry.get("file", {})
        file_name = file_info.get("file_name")
        name = file_info.get("name", key)

        if not file_name:
            continue

        path = os.path.join(scripts_base, f"{file_name}.txt")
        if not os.path.exists(path):
            missing += 1
            continue

        text = _read_script(path)
        if not text.strip():
            missing += 1
            continue

        movie_keys.append(key)
        movie_names.append(name)
        tokenized.append(tokenize(text))

    logger.info("BM25 index: %d documents (%d scripts not found)", len(movie_keys), missing)
    return movie_keys, movie_names, tokenized


class BM25Retriever:
    """BM25 retriever over movie scripts"""
    name = "bm25"

    def __init__(self, metadata_path: str = METADATA_PATH, scripts_base: str = SCRIPTS_BASE_PATH):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.movie_keys, self.movie_names, corpus = _build_corpus(metadata, scripts_base)
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "rank": rank + 1,
                "movie_key": self.movie_keys[idx],
                "movie_name": self.movie_names[idx],
                "score": float(scores[idx]),
            }
            for rank, (idx, _) in enumerate(ranked)
        ]
