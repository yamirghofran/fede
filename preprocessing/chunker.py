"""Script parsing and chunking for FEDE semantic indexing.

This module converts tagged screenplay files into structured chunks
(scenes and sentences) that can be embedded and indexed in Qdrant.
The tagged format uses single-letter prefixes:
    M: - Metadata
    S: - Scene start/heading
    N: - Scene description/narrative
    C: - Character name
    E: - Dialogue extension (e.g., V.O., O.S.)
    D: - Dialogue text
    T: - Transition
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from vector_db.indexer import SceneRecord, SentenceRecord
from vector_db.schemas import LineType


@dataclass
class SceneChunk:
    """Pre-embedding intermediate representation of a scene."""

    movie_id: str
    movie_title: str
    scene_id: str
    scene_index: int
    text: str  # scene_title + N: + D: (attributed) + T: joined by "\n"
    scene_title: Optional[str] = None
    character_names: Optional[List[str]] = None


@dataclass
class SentenceChunk:
    """Pre-embedding intermediate representation of a single line."""

    movie_id: str
    movie_title: str
    scene_id: str
    scene_index: int
    text: str  # raw line text (dialogue has NO character prefix)
    line_type: LineType  # "dialogue" | "description" | "transition"
    position_in_script: int  # global counter, increments on N:/D:/T: only
    character_name: Optional[str] = None  # set only for "dialogue"


class ScriptChunker:
    """Parses a tagged script file into scene and sentence chunks.

    Chunks are pre-embedding intermediates; call ``to_scene_records`` /
    ``to_sentence_records`` after computing embeddings to get objects
    compatible with ``ScriptIndexer.index_movie_batch``.
    """

    def __init__(self, movie_name: str, tagged_path: str) -> None:
        self.movie_title = movie_name
        self.tagged_path = tagged_path
        self.movie_id = movie_name.lower().replace(" ", "_").replace("-", "_")
        self._scene_chunks: Optional[List[SceneChunk]] = None
        self._sentence_chunks: Optional[List[SentenceChunk]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse(self) -> Tuple[List[SceneChunk], List[SentenceChunk]]:
        """Parse the tagged script file and cache results.

        Returns (scene_chunks, sentence_chunks).  Subsequent calls return
        the cached result without re-reading the file.
        """
        if self._scene_chunks is not None:
            return self._scene_chunks, self._sentence_chunks

        scene_chunks: List[SceneChunk] = []
        sentence_chunks: List[SentenceChunk] = []
        position = 0  # global counter incremented on N: / D: / T: only

        acc = {
            "index": -1,       # -1 = preamble (before first S:)
            "title": None,
            "lines": [],       # text fragments for scene.text
            "chars": [],       # ordered, deduplicated character names
            "cur_char": None,  # last seen C: value
            "cur_ext": None,   # last seen E: value (cleared after next D:)
        }

        with open(self.tagged_path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                tag, sep, content = raw.strip().partition(": ")
                if not sep:
                    continue

                if tag == "M":
                    # Metadata
                    pass  # skip metadata

                elif tag == "S":
                    # Start of a scene
                    if acc["index"] >= 0:
                        self._flush(acc, scene_chunks)
                    acc["index"] += 1
                    acc["title"] = content
                    acc["lines"] = [content]
                    acc["chars"] = []
                    acc["cur_char"] = None
                    acc["cur_ext"] = None

                elif tag == "N":
                    # Scene description
                    acc["lines"].append(content)
                    sentence_chunks.append(SentenceChunk(
                        movie_id=self.movie_id,
                        movie_title=self.movie_title,
                        scene_id=f"scene_{acc['index']:04d}",
                        scene_index=acc["index"],
                        text=content,
                        line_type="description",
                        position_in_script=position,
                        character_name=None,
                    ))
                    position += 1

                elif tag == "C":
                    # Character name
                    acc["cur_char"] = content
                    acc["cur_ext"] = None

                elif tag == "E":
                    # Dialogue metadata
                    acc["cur_ext"] = content

                elif tag == "D":
                    # Dialogue
                    cur_char = acc["cur_char"]
                    cur_ext = acc["cur_ext"]
                    if cur_ext:
                        attribution = f"{cur_char} ({cur_ext}): {content}"
                    else:
                        attribution = f"{cur_char}: {content}"
                    acc["lines"].append(attribution)
                    if cur_char and cur_char not in acc["chars"]:
                        acc["chars"].append(cur_char)
                    sentence_chunks.append(SentenceChunk(
                        movie_id=self.movie_id,
                        movie_title=self.movie_title,
                        scene_id=f"scene_{acc['index']:04d}",
                        scene_index=acc["index"],
                        text=content,
                        line_type="dialogue",
                        position_in_script=position,
                        character_name=cur_char,
                    ))
                    acc["cur_ext"] = None
                    position += 1

                elif tag == "T":
                    # Transition
                    acc["lines"].append(content)
                    sentence_chunks.append(SentenceChunk(
                        movie_id=self.movie_id,
                        movie_title=self.movie_title,
                        scene_id=f"scene_{acc['index']:04d}",
                        scene_index=acc["index"],
                        text=content,
                        line_type="transition",
                        position_in_script=position,
                        character_name=None,
                    ))
                    position += 1

        # flush final scene
        if acc["index"] >= 0:
            self._flush(acc, scene_chunks)

        self._scene_chunks = scene_chunks
        self._sentence_chunks = sentence_chunks
        return scene_chunks, sentence_chunks

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def scene_chunks(self) -> List[SceneChunk]:
        if self._scene_chunks is None:
            self.parse()
        return self._scene_chunks

    @property
    def sentence_chunks(self) -> List[SentenceChunk]:
        if self._sentence_chunks is None:
            self.parse()
        return self._sentence_chunks

    # ------------------------------------------------------------------
    # Conversion to indexer records
    # ------------------------------------------------------------------

    def to_scene_records(self, embeddings: List[List[float]]) -> List[SceneRecord]:
        """Pair scene chunks with their embeddings to produce ``SceneRecord`` objects."""
        chunks = self.scene_chunks
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Expected {len(chunks)} embeddings for scene chunks, got {len(embeddings)}"
            )
        return [
            SceneRecord(
                movie_id=c.movie_id,
                movie_title=c.movie_title,
                scene_id=c.scene_id,
                scene_index=c.scene_index,
                text=c.text,
                embedding=emb,
                scene_title=c.scene_title,
                character_names=c.character_names,
            )
            for c, emb in zip(chunks, embeddings)
        ]

    def to_sentence_records(self, embeddings: List[List[float]]) -> List[SentenceRecord]:
        """Pair sentence chunks with their embeddings to produce ``SentenceRecord`` objects."""
        chunks = self.sentence_chunks
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Expected {len(chunks)} embeddings for sentence chunks, got {len(embeddings)}"
            )
        return [
            SentenceRecord(
                movie_id=c.movie_id,
                movie_title=c.movie_title,
                scene_id=c.scene_id,
                scene_index=c.scene_index,
                text=c.text,
                line_type=c.line_type,
                position_in_script=c.position_in_script,
                embedding=emb,
                character_name=c.character_name,
            )
            for c, emb in zip(chunks, embeddings)
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flush(self, acc: dict, scene_chunks: List[SceneChunk]) -> None:
        scene_id = f"scene_{acc['index']:04d}"
        scene_chunks.append(SceneChunk(
            movie_id=self.movie_id,
            movie_title=self.movie_title,
            scene_id=scene_id,
            scene_index=acc["index"],
            text="\n".join(acc["lines"]),
            scene_title=acc["title"],
            character_names=list(acc["chars"]),
        ))
