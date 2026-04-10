import json
import os
import time
import logging
from typing import List, Optional, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints for resumable generation.
    Saves progress periodically and can resume from last checkpoint.
    """

    def __init__(self, checkpoint_path: str, checkpoint_interval: int = 50):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint file
            checkpoint_interval: Number of queries between checkpoints
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = os.path.dirname(checkpoint_path)

    def save_checkpoint(self, queries: List[Dict], completed: int, total: int):
        """
        Save current progress to checkpoint file.
        Atomic write to prevent corruption.

        Args:
            queries: List of generated queries so far
            completed: Number of queries completed
            total: Total number of queries to generate
        """
        checkpoint_data = {
            "queries": queries,
            "progress": {
                "completed": completed,
                "total": total,
                "percentage": (completed / total * 100) if total > 0 else 0,
            },
            "metadata": {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "checkpoint_version": "1.0",
            },
        }

        # Create checkpoint directory if needed
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Atomic write: write to temp file first
        temp_path = f"{self.checkpoint_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        # Rename to final path (atomic operation)
        os.rename(temp_path, self.checkpoint_path)

        logger.info(
            f"Checkpoint saved: {completed}/{total} queries ({completed / total * 100:.1f}%)"
        )

    def load_checkpoint(self) -> Optional[Dict]:
        """
        Load checkpoint if exists.

        Returns:
            Checkpoint data dict or None if no checkpoint exists
        """
        if not os.path.exists(self.checkpoint_path):
            logger.info("No existing checkpoint found")
            return None

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            completed = checkpoint_data["progress"]["completed"]
            total = checkpoint_data["progress"]["total"]

            logger.info(
                f"Checkpoint loaded: {completed}/{total} queries ({completed / total * 100:.1f}%)"
            )

            return checkpoint_data

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None

    def has_checkpoint(self) -> bool:
        """Check if checkpoint file exists."""
        return os.path.exists(self.checkpoint_path)

    def should_checkpoint(self, completed: int) -> bool:
        """
        Check if it's time to save a checkpoint.

        Args:
            completed: Number of queries completed so far

        Returns:
            True if should checkpoint, False otherwise
        """
        return completed > 0 and completed % self.checkpoint_interval == 0

    def clear_checkpoint(self):
        """Remove checkpoint file after successful completion."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            logger.info("Checkpoint cleared")
