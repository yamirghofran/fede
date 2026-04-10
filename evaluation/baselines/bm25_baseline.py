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

# Scene extraction settings - same ranges used in generate_scene_queries.py
SCENE_MIN_CHARS = 200
SCENE_MAX_CHARS = 3000

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
        except OSError as e:
            logger.warning("Error reading script at %s: %s", path, e)
            return ""
    logger.warning("Could not decode script %s with any of the attempted encodings.", path)
    return ""


def _extract_scenes(script_text: str) -> List[str]:
    """Split a screenplay into individual scenes.

    Uses INT./EXT. headings as primary boundaries.  Falls back to CUT TO:
    markers, then to fixed 1000-character chunks when no headings are found.
    Returns only scenes within the configured character-length window.
    """
    pattern = re.compile(
        r"(?:^|\n)[ \t]*(?:INT|EXT|INT\./EXT|I/E)[\./]",
        re.IGNORECASE,
    )
    boundaries = [m.start() for m in pattern.finditer(script_text)]

    if len(boundaries) < 3:
        boundaries = [m.start() for m in re.finditer(r"CUT TO:", script_text, re.IGNORECASE)]

    if len(boundaries) < 3:
        boundaries = list(range(0, len(script_text), 1000))

    scenes = [
        script_text[boundaries[i]: boundaries[i + 1]].strip()
        for i in range(len(boundaries) - 1)
    ]
    return [s for s in scenes if SCENE_MIN_CHARS <= len(s) <= SCENE_MAX_CHARS]


def _build_scene_corpus(
    metadata: Dict, scripts_base: str
) -> Tuple[List[str], List[str], List[List[str]]]:
    """Build a scene-level corpus.

    Returns:
        scene_movie_keys: movie key for each scene document
        scene_movie_names: movie display name for each scene document
        tokenized: tokenized scene text for each scene document
    """
    scene_movie_keys: List[str] = []
    scene_movie_names: List[str] = []
    tokenized: List[List[str]] = []
    missing = 0
    total_scenes = 0

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

        scenes = _extract_scenes(text)
        if not scenes:
            # Fallback: index the whole script as one document
            scenes = [text]

        for scene in scenes:
            scene_movie_keys.append(key)
            scene_movie_names.append(name)
            tokenized.append(tokenize(scene))
            total_scenes += 1

    logger.info(
        "BM25 scene index: %d scene documents from %d movies (%d scripts not found)",
        total_scenes,
        len(set(scene_movie_keys)),
        missing,
    )
    return scene_movie_keys, scene_movie_names, tokenized


class BM25Retriever:
    """Scene-level BM25 retriever with max-pool aggregation to movie level.

    Mirrors the ScenePoolEvaluator approach from finetuning/evaluation/scene_evaluator.py:
    each scene is a separate BM25 document; for a query the best-scoring scene
    per movie determines that movie's rank.
    """

    name = "bm25"

    def __init__(self, metadata_path: str = METADATA_PATH, scripts_base: str = SCRIPTS_BASE_PATH):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.scene_movie_keys, self.scene_movie_names, corpus = _build_scene_corpus(
            metadata, scripts_base
        )
        self.bm25 = BM25Okapi(corpus)
        # unique ordered list of movie keys (for external consumers that want the count)
        self.movie_keys = list(dict.fromkeys(self.scene_movie_keys))

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)

        # Max-pool: keep the best scene score per movie
        best: Dict[str, float] = {}
        names: Dict[str, str] = {}
        for idx, score in enumerate(scores):
            mkey = self.scene_movie_keys[idx]
            if score > best.get(mkey, float("-inf")):
                best[mkey] = score
                names[mkey] = self.scene_movie_names[idx]

        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "rank": rank + 1,
                "movie_key": k,
                "movie_name": names[k],
                "score": float(s),
            }
            for rank, (k, s) in enumerate(ranked)
        ]
