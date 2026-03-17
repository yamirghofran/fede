"""Collection names and typed payload schemas for FEDE's Qdrant collections."""

from enum import Enum
from typing import List, Literal, Optional

from typing_extensions import TypedDict


class CollectionNames(str, Enum):
    """Qdrant collection identifiers for FEDE."""

    SCENES = "scenes"
    SENTENCES = "sentences"


# Valid content classifications produced during script preprocessing.
LineType = Literal["dialogue", "description", "transition"]


class SentencePayload(TypedDict, total=False):
    """Payload stored alongside every sentence-level vector.

    Hierarchy: sentence → scene → movie.

    Fields marked Optional are populated when available from the preprocessed
    script but may be absent for non-dialogue lines (e.g. scene descriptions
    have no character_name).
    """

    # --- identity ---
    movie_id: str
    """Unique identifier for the source movie (e.g. TMDB ID as string)."""

    movie_title: str
    """Human-readable movie title, used for display and result grouping."""

    scene_id: str
    """Unique identifier for the parent scene, used for hierarchical resolution."""

    scene_index: int
    """Zero-based position of the parent scene within the script, used for
    temporal ordering of results."""

    # --- content ---
    text: str
    """Raw text of this sentence / line as it appears in the script."""

    line_type: LineType
    """Content classification: 'dialogue', 'description', or 'transition'."""

    character_name: Optional[str]
    """Speaking character for dialogue lines; None for non-dialogue lines."""

    position_in_script: int
    """Absolute zero-based line index within the full script, enabling
    fine-grained temporal ordering across scenes."""


class ScenePayload(TypedDict, total=False):
    """Payload stored alongside every scene-level vector.

    A scene vector represents the aggregated semantic content of one scene
    and serves as the primary retrieval unit returned to the user.
    """

    # --- identity ---
    movie_id: str
    """Unique identifier for the source movie (e.g. TMDB ID as string)."""

    movie_title: str
    """Human-readable movie title, used for display and result grouping."""

    scene_id: str
    """Unique identifier for this scene, referenced by its child sentence
    vectors for hierarchical resolution."""

    scene_index: int
    """Zero-based position of this scene within the script."""

    # --- content ---
    text: str
    """Full concatenated text of the scene (descriptions + dialogue)."""

    scene_title: Optional[str]
    """Scene heading / slug line if present (e.g. 'INT. COFFEE SHOP - DAY')."""

    character_names: List[str]
    """Deduplicated list of characters appearing in this scene, used for
    structural filtering and display."""
