#!/usr/bin/env python3
"""
Generate movie description queries using Google Gemini API.
Following methodology from Mustafa et al. [4] - IJECE Vol. 14, No. 6
Uses FULL movie scripts as input (no truncation).

Features:
- Checkpoint/Resume capability
- Lexical leakage detection
- Partial result backup
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.generator import GeminiQueryGenerator
from evaluation.config import (
    TARGET_NUM_QUERIES,
    MODEL_NAME,
    OUTPUT_PATH,
    CHECKPOINT_PATH,
    CHECKPOINT_INTERVAL,
    ENABLE_VALIDATION,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation queries for movie search using Google Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 300 queries with full scripts (default)
  python scripts/generate_queries.py

  # Generate 200 queries
  python scripts/generate_queries.py --num-queries 200

  # Resume from checkpoint
  python scripts/generate_queries.py --resume

  # Start fresh (ignore checkpoint)
  python scripts/generate_queries.py --no-resume

  # Use gemini-1.5-flash instead of pro
  python scripts/generate_queries.py --model gemini-1.5-flash

  # Disable validation for faster generation
  python scripts/generate_queries.py --no-validation
        """,
    )

    parser.add_argument(
        "--num-queries",
        type=int,
        default=TARGET_NUM_QUERIES,
        help=f"Number of queries to generate (default: {TARGET_NUM_QUERIES})",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        choices=["gemini-1.5-pro", "gemini-1.5-flash"],
        help=f"Gemini model to use (default: {MODEL_NAME})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help=f"Output file path (default: {OUTPUT_PATH})",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing checkpoint"
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoint",
    )

    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable lexical leakage validation",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N queries (default: {CHECKPOINT_INTERVAL})",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file")
        print("Please create a .env file with your Gemini API key:")
        print("  GEMINI_API_KEY=your-gemini-api-key-here")
        print()
        print("Get your free API key from: https://ai.google.dev/")
        sys.exit(1)

    # Check for existing checkpoint
    checkpoint_exists = os.path.exists(CHECKPOINT_PATH)

    if checkpoint_exists and not args.no_resume and not args.resume:
        print("=" * 70)
        print("CHECKPOINT DETECTED")
        print("=" * 70)
        print(f"Found existing checkpoint: {CHECKPOINT_PATH}")
        print()
        print("Options:")
        print(
            "  1. Resume from checkpoint: python scripts/generate_queries.py --resume"
        )
        print(
            "  2. Start fresh (delete checkpoint): python scripts/generate_queries.py --no-resume"
        )
        print()
        sys.exit(1)

    print("=" * 70)
    print("Movie Query Generator - Mustafa et al. [4] Methodology")
    print("Provider: Google Gemini API (gemini-1.5-pro - FREE TIER)")
    print("=" * 70)
    print(f"Number of queries: {args.num_queries}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Input: FULL scripts (no truncation)")
    print(f"Checkpointing: Enabled (every {args.checkpoint_interval} queries)")
    print(f"Validation: {'Enabled' if not args.no_validation else 'Disabled'}")
    print(
        f"Resume: {'Yes' if (args.resume or (checkpoint_exists and not args.no_resume)) else 'No'}"
    )
    print("=" * 70)

    # Calculate estimated time
    estimated_time_minutes = args.num_queries * 9 / 60
    print(f"Estimated time: {estimated_time_minutes:.0f} minutes")
    print()

    # Initialize generator
    print("Initializing Gemini generator...")
    generator = GeminiQueryGenerator(
        api_key=api_key,
        model=args.model,
        checkpoint_interval=args.checkpoint_interval,
        enable_validation=not args.no_validation,
    )

    # Generate queries
    print("Generating queries (this will take 45-60 minutes)...")
    print("Rate limiting: 15 requests per minute to stay within free tier")
    print("Features: Checkpointing, Validation, Partial Backup")
    print()
    queries = generator.generate_batch(
        num_queries=args.num_queries,
        resume=args.resume or (checkpoint_exists and not args.no_resume),
    )

    # Save results
    print()
    print("Saving results...")
    generator.save_queries(queries, args.output)

    # Clear checkpoint if successful
    if len(queries) >= args.num_queries:
        generator.checkpoint_manager.clear_checkpoint()

    print()
    print("=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"Total queries generated: {len(queries)}")
    print(f"Output saved to: {args.output}")
    print(f"Methodology: Mustafa et al. [4]")
    print(f"Input type: Full scripts")
    print(f"API used: Google Gemini ({args.model})")
    print(f"Cost: $0.00 (within free tier)")
    print(f"Validation: {'Enabled' if not args.no_validation else 'Disabled'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
