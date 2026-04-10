"""
Generate scene-level evaluation queries using OpenRouter.

Instead of describing the whole movie, the LLM is given a single scene
extracted from the script and asked to describe what happens - no character
names, no title. This aligns with how our embeddings are built (scene-level).

Usage:
    python scripts/generate_scene_queries.py
    python scripts/generate_scene_queries.py --num-queries 300 --resume
    python scripts/generate_scene_queries.py --num-queries 100 --no-resume
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, RateLimitError, APIConnectionError
from tqdm import tqdm

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from evaluation.dataset_generation.validator import QueryValidator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Paths
METADATA_PATH    = os.path.join(project_root, "data", "scripts", "metadata", "clean_parsed_meta.json")
SCRIPTS_BASE     = os.path.join(project_root, "data", "scripts", "unprocessed")
OUTPUT_PATH      = os.path.join(project_root, "evaluation", "evaluation_dataset", "generated_queries_scene.json")
CHECKPOINT_PATH  = os.path.join(project_root, "evaluation", "evaluation_dataset", "scene_queries_checkpoint.json")

# Generation settings
TARGET_NUM_QUERIES  = 300
CHECKPOINT_INTERVAL = 25
MAX_RETRIES         = 3
RATE_LIMIT_DELAY    = 1.5   # seconds between calls (OpenRouter is generous)

# Scene extraction: min/max chars for a scene to be usable
SCENE_MIN_CHARS = 300
SCENE_MAX_CHARS = 2500

# Prompt
SCENE_PROMPT = """\
Below is a single scene from a movie script.

{scene_text}

Write ONE sentence describing what happens in this scene. Requirements:
- Do NOT mention any character names (replace with roles: "a detective", "the woman", "two men", etc.)
- Do NOT mention the movie title
- Do NOT mention actor names, director, or any production details
- Describe the action, setting, and emotional tone of this specific scene
- The sentence should be specific enough that someone who saw the movie could recognise the scene
- Write in present tense
- Be concrete - avoid vague generalisations about themes

Output only the sentence, nothing else."""


def _extract_scenes(script_text: str) -> List[str]:
    """Split screenplay into individual scenes using INT./EXT. headings."""
    # Scene headings: INT., EXT., INT/EXT, I/E - optionally with whitespace
    pattern = re.compile(
        r'(?:^|\n)[ \t]*(?:INT|EXT|INT\./EXT|I/E)[\./]',
        re.IGNORECASE
    )
    boundaries = [m.start() for m in pattern.finditer(script_text)]

    if len(boundaries) < 3:
        # Fallback: split on CUT TO: / FADE IN / FADE OUT
        pattern2 = re.compile(r'\n\s*(?:CUT TO:|FADE (?:IN|OUT)|DISSOLVE TO:)', re.IGNORECASE)
        boundaries = [m.start() for m in pattern2.finditer(script_text)]

    if len(boundaries) < 3:
        # Last resort: split into ~800-char chunks
        chunks = []
        for i in range(0, len(script_text), 800):
            chunks.append(script_text[i:i + 800])
        return [c for c in chunks if SCENE_MIN_CHARS <= len(c) <= SCENE_MAX_CHARS]

    scenes = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(script_text)
        chunk = script_text[start:end].strip()
        if SCENE_MIN_CHARS <= len(chunk) <= SCENE_MAX_CHARS:
            scenes.append(chunk)

    return scenes


def _pick_scene(script_text: str, rng: random.Random) -> Optional[str]:
    """Pick a random usable scene from the script."""
    scenes = _extract_scenes(script_text)
    if not scenes:
        return None
    return rng.choice(scenes)


def _read_script(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            return f.read()


class SceneQueryGenerator:
    def __init__(self, api_key: str, api_url: str, model: str, enable_validation: bool = True):
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.model  = model
        self.metadata = json.load(open(METADATA_PATH, encoding="utf-8"))
        self.validator = QueryValidator(self.metadata) if enable_validation else None
        self.rng = random.Random(42)

    def _call_llm(self, scene_text: str) -> Optional[str]:
        prompt = SCENE_PROMPT.format(scene_text=scene_text[:SCENE_MAX_CHARS])
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=200,
                )
                if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                    return resp.choices[0].message.content.strip()
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                wait = 30 if "429" in str(e) or "rate" in str(e).lower() else 2 ** attempt
                logger.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {e}. Waiting {wait}s.")
                time.sleep(wait)
        return None

    def generate_one(self, movie_key: str, query_id: int) -> Optional[Dict]:
        entry    = self.metadata[movie_key]
        file_info = entry.get("file", {})
        tmdb_info = entry.get("tmdb", {})
        file_name = file_info.get("file_name")
        source    = file_info.get("source")
        script_path = os.path.join(SCRIPTS_BASE, source, f"{file_name}.txt")

        if not os.path.exists(script_path):
            return None

        script_text = _read_script(script_path)
        scene = _pick_scene(script_text, self.rng)
        if not scene:
            return None

        query = self._call_llm(scene)
        if not query:
            return None

        result = {
            "id": query_id,
            "query": query,
            "movie_name": file_info.get("name"),
            "movie_key": movie_key,
            "metadata": {
                "release_date": tmdb_info.get("release_date"),
                "tmdb_id": tmdb_info.get("id"),
                "source": source,
                "file_name": file_name,
                "scene_chars": len(scene),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "model": self.model,
                "query_type": "scene_level",
            },
        }

        if self.validator:
            result["validation"] = self.validator.check_lexical_leakage(
                query, file_info.get("name", "")
            )

        return result

    def generate_batch(self, num_queries: int, resume: bool = False) -> List[Dict]:
        results: List[Dict] = []
        start_id = 0

        if resume and os.path.exists(CHECKPOINT_PATH):
            ckpt = json.load(open(CHECKPOINT_PATH, encoding="utf-8"))
            results  = ckpt.get("queries", [])
            start_id = len(results)
            print(f"Resuming from query {start_id + 1}")

        processed_keys = {r["movie_key"] for r in results}
        all_keys       = list(self.metadata.keys())
        remaining_keys = [k for k in all_keys if k not in processed_keys]
        self.rng.shuffle(remaining_keys)
        keys_to_use = remaining_keys[:num_queries - start_id]

        failed = 0
        for i, movie_key in enumerate(tqdm(keys_to_use, desc="Generating scene queries")):
            query_id = start_id + i + 1
            result = self.generate_one(movie_key, query_id)
            if result:
                results.append(result)
            else:
                failed += 1

            # Checkpoint
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
                with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                    json.dump({"queries": results}, f, ensure_ascii=False)

            if i < len(keys_to_use) - 1:
                time.sleep(RATE_LIMIT_DELAY)

        print(f"Generated {len(results)} queries, failed: {failed}")
        return results

    def save(self, queries: List[Dict], output_path: str):
        validation_report = None
        if self.validator:
            validation_report = self.validator.validate_batch(queries)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"evaluation_queries": queries, "validation_report": validation_report},
                f, indent=2, ensure_ascii=False,
            )
        print(f"Saved {len(queries)} queries -> {output_path}")

        if validation_report:
            print(f"\nValidation: {validation_report['passed']}/{validation_report['total']} passed, "
                  f"{validation_report['flagged']} flagged for leakage "
                  f"(avg score {validation_report['avg_leakage_score']:.3f})")
            if validation_report["flagged_queries"]:
                print("\nFlagged queries:")
                for fq in validation_report["flagged_queries"][:10]:
                    print(f"  ID {fq['id']} [{fq['movie_name']}]: {fq['reason']} (score {fq['leakage_score']:.2f})")
                    print(f"    {fq['query'][:120]}")


def main():
    parser = argparse.ArgumentParser(description="Generate scene-level evaluation queries via OpenRouter")
    parser.add_argument("--num-queries",  type=int,  default=TARGET_NUM_QUERIES)
    parser.add_argument("--output",       type=str,  default=OUTPUT_PATH)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--no-resume",    action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    model   = os.getenv("LLM_MODEL",   "google/gemini-2.0-flash-001")

    if not api_key:
        print("ERROR: LLM_API_KEY not set in .env")
        sys.exit(1)

    resume = args.resume and not args.no_resume

    print("=" * 60)
    print("FEDE - Scene Query Generation (OpenRouter)")
    print("=" * 60)
    print(f"Model         : {model}")
    print(f"Queries       : {args.num_queries}")
    print(f"Output        : {args.output}")
    print(f"Resume        : {resume}")
    print(f"Validation    : {not args.no_validation}")
    print("=" * 60)

    gen = SceneQueryGenerator(
        api_key=api_key,
        api_url=api_url,
        model=model,
        enable_validation=not args.no_validation,
    )

    queries = gen.generate_batch(args.num_queries, resume=resume)
    gen.save(queries, args.output)

    # Clean checkpoint if done
    if len(queries) >= args.num_queries and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
