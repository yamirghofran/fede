"""Configuration for evaluation query generation."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data paths
METADATA_PATH = os.path.join(BASE_DIR, "data", "scripts", "metadata", "clean_parsed_meta.json")
RELATIONS_DIR = os.path.join(BASE_DIR, "data", "knowledge_graph", "relations")
EVAL_QUERIES_PATH = os.path.join(BASE_DIR, "evaluation", "evaluation_dataset", "eval_queries.json")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "evaluation", "evaluation_dataset", "gen_checkpoint.json")

# LLM settings
# Reads LLM_API_KEY and LLM_MODEL from .env (same as the rest of the project)
LLM_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"

# Generation settings
QUERIES_PER_MOVIE = 3 # default number of queries to generate per movie
MIN_RELATIONS = 10# skip movies with fewer relations than this
CHECKPOINT_INTERVAL = 5 # save checkpoint every N movies
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.2 # seconds between API calls

# Validation settings
VALIDATION_STRICTNESS = "medium"

# Sentinel: _scene_idx >= EXTENDED_IDX_BASE marks queries added by this generator
EXTENDED_IDX_BASE = 100

# LLM prompt
GENERATION_PROMPT = """\
Below are excerpts describing events and interactions between characters in a movie.

Movie title: {movie_title}

Excerpts:
{relations_text}

Write {n} search queries that a person might type when trying to find this movie based on its story.
Each query should describe what happens between specific characters - their actions, conflicts, relationships, or decisions.

Rules:
- You MAY use character names
- Do NOT mention the movie title
- Each query should be 20-45 words
- Each query must be on its own line, no numbering or bullets
- Output only the queries, nothing else"""
