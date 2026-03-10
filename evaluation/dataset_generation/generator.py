"""
Movie Query Generator using OpenAI with Checkpointing and Validation
Following Mustafa et al. [4] methodology - IJECE Vol. 14, No. 6
Uses FULL movie scripts as input (no truncation)
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from evaluation.dataset_generation.checkpoint_manager import CheckpointManager
from evaluation.dataset_generation.config import (
    BACKUP_AFTER_EACH_QUERY,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PATH,
    MAX_RETRIES,
    METADATA_PATH,
    MODEL_NAME,
    OUTPUT_PATH,
    RATE_LIMIT_DELAY,
    RETRY_DELAY,
    SCRIPTS_BASE_PATH,
    SYSTEM_PROMPT,
)
from evaluation.dataset_generation.validator import QueryValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIQueryGenerator:
    """
    Generates movie description queries using OpenAI.
    Features: Checkpointing, Validation, Backup.
    """

    def __init__(
        self,
        api_key: str,
        model: str = MODEL_NAME,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        enable_validation: bool = True,
    ):
        """
        Initialize OpenAI generator with all features.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
            checkpoint_interval: Save checkpoint every N queries
            enable_validation: Enable lexical leakage detection
        """
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model = model
        self.metadata = self._load_metadata()

        # Initialize validator
        self.validator = QueryValidator(self.metadata) if enable_validation else None

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            CHECKPOINT_PATH, checkpoint_interval
        )

    def _load_metadata(self) -> Dict:
        """Load movie metadata from clean_parsed_meta.json"""
        metadata_path = Path(METADATA_PATH)
        if not metadata_path.exists():
            # Try alternative path
            alt_path = (
                Path(__file__).parent.parent
                / "data/scripts/metadata/clean_parsed_meta.json"
            )
            if alt_path.exists():
                metadata_path = alt_path

        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_script(self, script_path: str) -> str:
        """Read full movie script with encoding fallback."""
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(script_path, "r", encoding="latin-1") as f:
                return f.read()

    def _build_prompt(self, script_content: str) -> str:
        """Build prompt using FULL script content."""
        return f"""Movie script: {script_content}

{SYSTEM_PROMPT}"""

    def _generate_with_retry(
        self, prompt: str, max_retries: int = MAX_RETRIES
    ) -> Optional[str]:
        """Generate query with automatic retry on failure."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=1000,
                    top_p=0.8,
                )

                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    if choice.finish_reason == "content_filter":
                        logger.warning(
                            f"Response blocked by content filter (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(2**attempt)
                        continue

                    if choice.message and choice.message.content:
                        query = choice.message.content.strip()
                        return query
                    else:
                        logger.warning(
                            f"Empty response (attempt {attempt + 1}/{max_retries})"
                        )
                        if attempt < max_retries - 1:
                            time.sleep(2**attempt)
                        continue

            except api_exceptions.ResourceExhausted as e:
                logger.error(
                    f"Error during generation (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if (
                    "rate_limit" in error_str.lower()
                    or "limit" in error_str.lower()
                    or "429" in error_str
                ):
                    logger.warning("Rate limit hit, waiting 30 seconds...")
                    time.sleep(30)
                elif attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue

        logger.error("All retry attempts failed")
        return None

    def generate_query(self, movie_key: str, query_id: int) -> Optional[Dict]:
        """
        Generate a single query for a movie using FULL script.
        Includes validation.
        """
        movie_entry = self.metadata[movie_key]
        file_info = movie_entry.get("file", {})
        tmdb_info = movie_entry.get("tmdb", {})

        # Build script path
        file_name = file_info.get("file_name")
        source = file_info.get("source")
        script_path = os.path.join(SCRIPTS_BASE_PATH, source, f"{file_name}.txt")

        # Check if script exists
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return None

        # Read FULL script
        script_content = self._read_script(script_path)

        # Build prompt
        prompt = self._build_prompt(script_content)

        # Generate with retry
        query = self._generate_with_retry(prompt)

        if query:
            result = {
                "id": query_id,
                "query": query,
                "movie_name": file_info.get("name"),
                "movie_key": movie_key,
                "metadata": {
                    "release_date": tmdb_info.get("release_date"),
                    "tmdb_id": tmdb_info.get("id"),
                    "overview": tmdb_info.get("overview"),
                    "source": source,
                    "file_name": file_name,
                    "script_path": script_path,
                    "script_size_bytes": len(script_content),
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "model": self.model,
                },
            }

            # Validate query
            if self.validator:
                validation_result = self.validator.check_lexical_leakage(
                    query, file_info.get("name")
                )
                result["validation"] = validation_result

            return result

        return None

    def generate_batch(
        self, num_queries: int = 300, resume: bool = False
    ) -> List[Dict]:
        """
        Generate queries for multiple randomly sampled movies.
        Supports checkpoint/resume.

        Args:
            num_queries: Number of queries to generate
            resume: Whether to resume from checkpoint

        Returns:
            List of generated query dictionaries
        """
        # Check for existing checkpoint
        if resume:
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                results = checkpoint_data["queries"]
                completed = checkpoint_data["progress"]["completed"]

                if completed >= num_queries:
                    logger.info("Already completed all queries!")
                    return results

                logger.info(f"Resuming from query {completed + 1}")

                # Get remaining movies to process
                all_keys = list(self.metadata.keys())
                processed_keys = {r["movie_key"] for r in results}
                remaining_keys = [k for k in all_keys if k not in processed_keys]

                # Sample remaining
                needed = num_queries - completed
                sampled_keys = random.sample(
                    remaining_keys, min(needed, len(remaining_keys))
                )
                start_id = completed
            else:
                # No checkpoint, start fresh
                movie_keys = list(self.metadata.keys())
                sampled_keys = random.sample(
                    movie_keys, min(num_queries, len(movie_keys))
                )
                results = []
                start_id = 0
        else:
            # Fresh start
            movie_keys = list(self.metadata.keys())
            sampled_keys = random.sample(movie_keys, min(num_queries, len(movie_keys)))
            results = []
            start_id = 0

        logger.info(f"Generating {len(sampled_keys)} queries using {self.model}")
        logger.info(f"Method: Full scripts (no truncation)")
        logger.info(f"Features: Checkpointing, Validation, Backup")

        failed_count = 0

        for i, movie_key in enumerate(tqdm(sampled_keys, desc="Generating queries")):
            query_id = start_id + i + 1
            result = self.generate_query(movie_key, query_id)

            if result:
                results.append(result)
            else:
                failed_count += 1
                logger.warning(f"Failed to generate query for: {movie_key}")

            # Partial backup: save after each query
            if BACKUP_AFTER_EACH_QUERY:
                self._save_partial_backup(results, query_id, num_queries)

            # Checkpoint save
            completed = start_id + i + 1
            if self.checkpoint_manager.should_checkpoint(completed):
                self.checkpoint_manager.save_checkpoint(results, completed, num_queries)

            # Rate limiting
            if i < len(sampled_keys) - 1:
                time.sleep(RATE_LIMIT_DELAY)

        logger.info(
            f"Successfully generated {len(results)} queries, failed: {failed_count}"
        )

        return results

    def _save_partial_backup(self, queries: List[Dict], completed: int, total: int):
        """Save partial backup."""
        backup_path = OUTPUT_PATH.replace(".json", "_backup.json")

        # Ensure output directory exists before saving
        backup_dir = os.path.dirname(backup_path)
        os.makedirs(backup_dir, exist_ok=True)

        backup_data = {
            "evaluation_queries": queries,
            "dataset_info": {
                "total_queries": completed,
                "methodology": "Mustafa et al. [4] - IJECE Vol. 14, No. 6",
                "prompt": SYSTEM_PROMPT,
                "input_type": "full_scripts",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "version": "1.0",
                "model": self.model,
                "status": "partial_backup",
            },
        }

        try:
            # Atomic write
            temp_path = f"{backup_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            os.rename(temp_path, backup_path)
        except Exception as e:
            logger.warning(f"Failed to save partial backup: {e}")

    def save_queries(self, queries: List[Dict], output_path: str):
        """
        Save generated queries to JSON file with validation report.
        """
        # Generate validation report
        if self.validator:
            validation_report = self.validator.validate_batch(queries)
        else:
            validation_report = None

        output_data = {
            "evaluation_queries": queries,
            "validation_report": validation_report,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(queries)} queries to {output_path}")

        # Print validation summary
        if validation_report:
            print("\n" + "=" * 70)
            print("VALIDATION SUMMARY")
            print("=" * 70)
            print(f"Total queries: {validation_report['total']}")
            print(f"Passed validation: {validation_report['passed']}")
            print(f"Flagged for leakage: {validation_report['flagged']}")
            print(
                f"Average leakage score: {validation_report['avg_leakage_score']:.3f}"
            )
            print()

            if validation_report["flagged_queries"]:
                print("FLAGGED QUERIES (may need manual review):")
                print("-" * 70)
                for flagged in validation_report["flagged_queries"][
                    :10
                ]:  # Show first 10
                    print(f"ID {flagged['id']}: {flagged['movie_name']}")
                    print(f"  Query: {flagged['query']}")
                    print(f"  Reason: {flagged['reason']}")
                    print(f"  Score: {flagged['leakage_score']:.2f}")
                    print()

                if len(validation_report["flagged_queries"]) > 10:
                    print(
                        f"... and {len(validation_report['flagged_queries']) - 10} more"
                    )

            print("=" * 70)
