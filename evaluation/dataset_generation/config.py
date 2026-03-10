"""
Configuration for OpenAI query generation with validation and checkpointing
"""

import os

# OpenAI API Configuration
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
MODEL_NAME = "google/gemini-3.1-flash-lite-preview"

# Generation Settings
TARGET_NUM_QUERIES = 300
SAMPLE_STRATEGY = "random"

# Checkpoint Settings
CHECKPOINT_INTERVAL = 50
CHECKPOINT_FILE = "generated_queries_checkpoint.json"

# Validation Settings
ENABLE_VALIDATION = True
VALIDATION_STRICTNESS = "low"
MAX_LEAKAGE_SCORE = 0.7

# Data Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_PATH = os.path.join(
    BASE_DIR, "data", "scripts", "metadata", "clean_parsed_meta.json"
)
SCRIPTS_BASE_PATH = os.path.join(BASE_DIR, "data", "scripts", "unprocessed")
OUTPUT_PATH = os.path.join(BASE_DIR, "evaluation_dataset", "generated_queries.json")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "evaluation_dataset", CHECKPOINT_FILE)

# Prompt
SYSTEM_PROMPT = """You are generating evaluation queries for a movie search system. Generate a one sentence descriptive summary that captures the essence, plot, themes, or atmosphere of the movie.

Requirements:
- DO NOT mention the movie title
- DO NOT mention character names
- DO NOT mention actor names
- DO NOT include specific identifiers (dates, locations, proper nouns that could identify the movie)

Focus on:
- The main storyline or premise
- Key themes explored in the film
- Emotional tone or atmosphere
- Unique elements that make this movie memorable
- Genre or style of the film
- The central conflict or journey

The query should be a complete and grammatically correct sentence that would help someone identify this movie if they had seen it but couldn't remember the title.

Example of a good query: "A young orphan discovers he has magical powers and must attend a wizarding school to defeat a dark wizard who killed his parents."

"""

# Retry Settings
MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 60

# Rate Limit Handling
REQUESTS_PER_MINUTE = 15
RATE_LIMIT_DELAY = 4

# Backup Settings
BACKUP_ENABLED = True
BACKUP_AFTER_EACH_QUERY = True
