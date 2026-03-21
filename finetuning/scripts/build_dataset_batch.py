#!/usr/bin/env python3
"""CLI: Build the round-1 training dataset via Gemini Batch API.

Three modes, run in order::

    # 1. Build prompt JSONL + submit batch job
    python -m finetuning.scripts.build_dataset_batch submit --movies 1200

    # 2. Poll until complete (can re-run safely)
    python -m finetuning.scripts.build_dataset_batch poll --job batches/123456

    # 3. Download results + post-process into training pairs
    python -m finetuning.scripts.build_dataset_batch process --job batches/123456

Or do all three in one shot::

    python -m finetuning.scripts.build_dataset_batch all --movies 1200
"""

from __future__ import annotations

import argparse
import logging
import sys

from finetuning.config import FINETUNING_DATA_DIR, GEMINI_BATCH_MODEL
from finetuning.dataset.batch_builder import BatchDatasetBuilder

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build round-1 finetuning dataset via Gemini Batch API",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- submit ---
    p_submit = sub.add_parser("submit", help="Build prompts and submit batch job")
    p_submit.add_argument("--movies", type=int, default=None, help="Max movies (default: all)")
    p_submit.add_argument("--model", type=str, default=GEMINI_BATCH_MODEL, help="Gemini model")

    # --- poll ---
    p_poll = sub.add_parser("poll", help="Poll batch job until done")
    p_poll.add_argument("--job", type=str, required=True, help="Batch job name (e.g. batches/123456)")
    p_poll.add_argument("--interval", type=int, default=30, help="Poll interval seconds")

    # --- process ---
    p_process = sub.add_parser("process", help="Download + process batch results")
    p_process.add_argument("--job", type=str, required=True, help="Batch job name")
    p_process.add_argument("--movies", type=int, default=None, help="Max movies (must match submit)")
    p_process.add_argument("--output", type=str, default=None, help="Output JSONL path")

    # --- all ---
    p_all = sub.add_parser("all", help="Submit → poll → process in one shot")
    p_all.add_argument("--movies", type=int, default=None, help="Max movies (default: all)")
    p_all.add_argument("--model", type=str, default=GEMINI_BATCH_MODEL, help="Gemini model")
    p_all.add_argument("--output", type=str, default=None, help="Output JSONL path")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.mode == "submit":
        builder = BatchDatasetBuilder(max_movies=args.movies, model=args.model)
        job_name = builder.build_and_submit()
        print(f"\nBatch job submitted: {job_name}")
        print(f"Next: python -m finetuning.scripts.build_dataset_batch poll --job {job_name}")

    elif args.mode == "poll":
        builder = BatchDatasetBuilder()
        state = builder.poll_until_done(args.job, poll_interval=args.interval)
        if state == "JOB_STATE_SUCCEEDED":
            print(f"\nJob succeeded. Next:")
            print(f"  python -m finetuning.scripts.build_dataset_batch process --job {args.job}")
        else:
            print(f"\nJob ended with state: {state}", file=sys.stderr)
            sys.exit(1)

    elif args.mode == "process":
        from pathlib import Path
        output = Path(args.output) if args.output else None
        builder = BatchDatasetBuilder(max_movies=args.movies)
        result = builder.process_results(job_name=args.job, output_path=output)
        print(f"\nTraining pairs written to: {result}")

    elif args.mode == "all":
        from pathlib import Path
        output = Path(args.output) if args.output else None
        builder = BatchDatasetBuilder(max_movies=args.movies, model=args.model)

        logger.info("=== Stage 1/3: Building prompts + submitting batch ===")
        job_name = builder.build_and_submit()
        print(f"Batch job: {job_name}")

        logger.info("=== Stage 2/3: Polling until complete ===")
        state = builder.poll_until_done(job_name)
        if state != "JOB_STATE_SUCCEEDED":
            print(f"Job failed with state: {state}", file=sys.stderr)
            sys.exit(1)

        logger.info("=== Stage 3/3: Processing results ===")
        result = builder.process_results(job_name=job_name, output_path=output)
        print(f"\nTraining pairs written to: {result}")


if __name__ == "__main__":
    main()
