"""Offline indexing pipeline: tagged scripts → embeddings → Qdrant.

Reads all tagged movie scripts from data/scripts/parsed/tagged/, chunks them
with ScriptChunker, embeds with a HuggingFace SentenceTransformer, and
bulk-upserts into Qdrant via ScriptIndexer.

Usage:
    python scripts/upload_scripts.py [options]

Options:
    --batch-size N   Embedding batch size (overrides EMBEDDING_BATCH_SIZE, default 64)
    --device DEVICE  Compute device: cpu | cuda | mps | auto (overrides EMBEDDING_DEVICE)
    --limit N        Process only the first N movies (useful for smoke-testing)
    --no-resume      Ignore the progress file and re-index all movies
    --workers N      Number of parallel worker processes (default 1)
                     Each worker loads its own model copy. Recommended: 2–4 on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve project root so the script can be run from any working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from preprocessing.chunker import ScriptChunker
from vector_db import ScriptIndexer, initialize_all_collections

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TAGGED_DIR = ROOT / "data" / "scripts" / "parsed" / "tagged"
METADATA_PATH = ROOT / "data" / "scripts" / "metadata" / "clean_parsed_meta.json"
PROGRESS_FILE = ROOT / "scripts" / ".indexed_movies.txt"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multiprocessing: lock injected into worker processes via Pool initializer
# ---------------------------------------------------------------------------
_progress_lock: Optional[mp.synchronize.Lock] = None


def _worker_init(lock: mp.synchronize.Lock) -> None:
    """Pool initializer — stores the shared lock and re-loads env in the worker."""
    global _progress_lock
    _progress_lock = lock
    # Prevent HuggingFace fast tokenizers from spinning up their own loky/joblib
    # thread pool inside each worker. Without this, semaphore objects leak at
    # shutdown because loky doesn't clean up when the parent process exits.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Workers are spawned fresh; re-insert project root and reload .env
    sys.path.insert(0, str(ROOT))
    load_dotenv(ROOT / ".env", override=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s][W%(process)d] %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_arg: str) -> str:
    """Return a torch device string from "auto" or explicit value."""
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_model(model_name: str, device: str) -> SentenceTransformer:
    log.info("Loading embedding model '%s' on device '%s' ...", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    log.info(
        "Model loaded — vector size: %d, max seq length: %d",
        model.get_sentence_embedding_dimension(),
        model.max_seq_length,
    )
    return model


def _encode(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> List[List[float]]:
    """Encode document texts for indexing; returns List[List[float]] ready for Qdrant.

    Uses model.encode_document() when available (EmbeddingGemma models register
    query/document prompts internally), falling back to plain encode(). This ensures
    indexed vectors are in the correct prompt space — never use bare encode() here
    and encode_queries() at search time, as they would produce mismatched spaces.
    """
    encode_fn = getattr(model, "encode_document", None)
    if callable(encode_fn):
        vecs = encode_fn(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    else:
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return np.asarray(vecs, dtype=np.float32).tolist()


def _load_progress(progress_file: Path) -> set[str]:
    if not progress_file.exists():
        return set()
    lines = progress_file.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def _mark_done(progress_file: Path, movie_id: str) -> None:
    """Append movie_id to the progress file, using a lock when running in parallel."""
    def _write() -> None:
        with progress_file.open("a", encoding="utf-8") as fh:
            fh.write(movie_id + "\n")

    if _progress_lock is not None:
        with _progress_lock:
            _write()
    else:
        _write()


# ---------------------------------------------------------------------------
# Per-movie processing
# ---------------------------------------------------------------------------

def _process_movie(
    *,
    meta_key: str,
    meta: dict,
    model: SentenceTransformer,
    indexer: ScriptIndexer,
    batch_size: int,
    progress_file: Path,
) -> bool:
    """Chunk, embed, and index a single movie.  Returns True on success."""
    movie_name: str = meta["file"]["name"]
    tagged_filename: Optional[str] = meta.get("parsed", {}).get("tagged")

    if not tagged_filename:
        log.warning("[%s] No tagged file entry in metadata — skipping.", meta_key)
        return False

    tagged_path = TAGGED_DIR / tagged_filename
    if not tagged_path.exists():
        log.warning("[%s] Tagged file not found: %s — skipping.", meta_key, tagged_path)
        return False

    try:
        chunker = ScriptChunker(movie_name=movie_name, tagged_path=str(tagged_path))
        scene_chunks, sentence_chunks = chunker.parse()
    except Exception as exc:
        log.error("[%s] Chunking failed: %s", meta_key, exc)
        return False

    if not scene_chunks:
        log.warning("[%s] No scenes found — skipping.", meta_key)
        return False

    try:
        scene_texts = [sc.text for sc in scene_chunks]
        sent_texts = [sc.text for sc in sentence_chunks]

        scene_embeddings = _encode(model, scene_texts, batch_size)
        sent_embeddings = _encode(model, sent_texts, batch_size) if sent_texts else []

        scene_records = chunker.to_scene_records(scene_embeddings)
        sent_records = chunker.to_sentence_records(sent_embeddings) if sent_texts else []

        indexer.index_movie_batch(scene_records, sent_records)
    except Exception as exc:
        log.error("[%s] Indexing failed: %s", meta_key, exc, exc_info=True)
        return False

    _mark_done(progress_file, chunker.movie_id)
    return True


# ---------------------------------------------------------------------------
# Parallel worker entry point
# ---------------------------------------------------------------------------

def _worker_fn(
    args: Tuple,
) -> Tuple[int, int]:
    """Process a chunk of movies in a worker process.

    Each worker loads its own model instance. PyTorch thread count is capped
    at ``threads_per_worker`` to avoid CPU over-subscription across workers.
    Qdrant upserts are safe to run concurrently — the server handles them, and
    all point IDs are deterministic + movie-scoped so there is no collision risk.
    """
    movies_chunk, model_name, device, batch_size, progress_file_str, threads_per_worker = args

    # Cap PyTorch intra-op threads to avoid N workers each fighting for all cores
    try:
        import torch
        torch.set_num_threads(max(1, threads_per_worker))
    except Exception:
        pass

    model = _load_model(model_name, device)
    # Each worker creates its own Qdrant client — thread-safe, no shared state
    indexer = ScriptIndexer()
    progress_file = Path(progress_file_str)

    succeeded = failed = 0
    for meta_key, meta in movies_chunk:
        ok = _process_movie(
            meta_key=meta_key,
            meta=meta,
            model=model,
            indexer=indexer,
            batch_size=batch_size,
            progress_file=progress_file,
        )
        if ok:
            succeeded += 1
        else:
            failed += 1

    log.info("Worker done — succeeded=%d failed=%d", succeeded, failed)
    return succeeded, failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Index movie scripts into Qdrant.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        help="Embedding batch size (default: EMBEDDING_BATCH_SIZE env or 64)",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("EMBEDDING_DEVICE", "auto"),
        help="Compute device: cpu | cuda | mps | auto (default: EMBEDDING_DEVICE env or auto)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N movies (for testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore the progress file and re-index all movies",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes (default: 1). "
            "Each worker loads its own model copy. "
            "Recommended: 2–4 on CPU with 24 GB RAM."
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        log.error("--workers must be >= 1")
        sys.exit(1)

    # --- Model name + device (resolved once; workers inherit these values) ---
    model_name = os.getenv("EMBEDDING_MODEL_ID")
    if not model_name:
        log.error(
            "EMBEDDING_MODEL_ID is not set. "
            "Add it to .env (e.g. EMBEDDING_MODEL_ID=sentence-transformers/all-mpnet-base-v2)"
        )
        sys.exit(1)

    device = _resolve_device(args.device)

    # Validate vector size by loading the model once in the main process
    model = _load_model(model_name, device)
    model_dim = model.get_sentence_embedding_dimension()
    qdrant_dim = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))
    if model_dim != qdrant_dim:
        log.error(
            "Model output dimension (%d) does not match QDRANT_VECTOR_SIZE (%d). "
            "Update QDRANT_VECTOR_SIZE in .env to match the model.",
            model_dim,
            qdrant_dim,
        )
        sys.exit(1)

    # --- Qdrant: initialise collections once before spawning workers ---
    log.info("Initialising Qdrant collections ...")
    initialize_all_collections()

    # --- Metadata ---
    log.info("Loading metadata from %s ...", METADATA_PATH)
    with METADATA_PATH.open(encoding="utf-8") as fh:
        all_meta: dict = json.load(fh)
    log.info("Found %d movies in metadata.", len(all_meta))

    # --- Resume ---
    already_indexed: set[str] = set()
    if not args.no_resume:
        already_indexed = _load_progress(PROGRESS_FILE)
        if already_indexed:
            log.info("Resuming — %d movies already indexed, skipping.", len(already_indexed))

    def _meta_key_to_movie_id(name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    movies_to_process = [
        (meta_key, meta)
        for meta_key, meta in all_meta.items()
        if _meta_key_to_movie_id(meta["file"]["name"]) not in already_indexed
    ]

    if args.limit is not None:
        movies_to_process = movies_to_process[: args.limit]

    log.info("%d movies to index.", len(movies_to_process))

    succeeded = failed = 0

    # -----------------------------------------------------------------------
    # Single-worker path (original behaviour, keeps tqdm progress bar)
    # -----------------------------------------------------------------------
    if args.workers == 1:
        indexer = ScriptIndexer()
        with tqdm(movies_to_process, unit="movie", dynamic_ncols=True) as pbar:
            for meta_key, meta in pbar:
                pbar.set_description(meta["file"]["name"][:40])
                ok = _process_movie(
                    meta_key=meta_key,
                    meta=meta,
                    model=model,
                    indexer=indexer,
                    batch_size=args.batch_size,
                    progress_file=PROGRESS_FILE,
                )
                if ok:
                    succeeded += 1
                else:
                    failed += 1

    # -----------------------------------------------------------------------
    # Multi-worker path
    # -----------------------------------------------------------------------
    else:
        # Release the main-process model before forking to avoid duplicating
        # its memory into every child (spawn still copies imports, but not tensors)
        del model

        n_workers = min(args.workers, len(movies_to_process))
        log.info("Spawning %d worker processes ...", n_workers)

        # Distribute movies as evenly as possible across workers
        chunk_size = math.ceil(len(movies_to_process) / n_workers)
        chunks = [
            movies_to_process[i : i + chunk_size]
            for i in range(0, len(movies_to_process), chunk_size)
        ]

        # How many PyTorch threads each worker should use
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1
        threads_per_worker = max(1, cpu_count // n_workers)

        worker_args = [
            (chunk, model_name, device, args.batch_size, str(PROGRESS_FILE), threads_per_worker)
            for chunk in chunks
        ]

        manager = mp.Manager()
        lock = manager.Lock()

        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(lock,),
        ) as pool:
            results = pool.map(_worker_fn, worker_args)

        for s, f in results:
            succeeded += s
            failed += f

    log.info(
        "Done. succeeded=%d  failed=%d  skipped(already indexed)=%d",
        succeeded,
        failed,
        len(already_indexed),
    )


if __name__ == "__main__":
    main()
