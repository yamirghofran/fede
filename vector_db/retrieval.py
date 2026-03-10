"""Online retrieval layer — semantic search over scenes and sentences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .client import get_qdrant_client
from .config import QdrantConfig
from .schemas import CollectionNames, LineType, ScenePayload, SentencePayload


@dataclass
class SentenceResult:
    """A single sentence retrieved from the sentences collection."""

    point_id: str
    score: float
    movie_id: str
    movie_title: str
    scene_id: str
    scene_index: int
    text: str
    line_type: LineType
    position_in_script: int
    character_name: Optional[str] = None


@dataclass
class SceneResult:
    """A single scene retrieved from the scenes collection."""

    point_id: str
    score: float
    movie_id: str
    movie_title: str
    scene_id: str
    scene_index: int
    text: str
    scene_title: Optional[str] = None
    character_names: List[str] = field(default_factory=list)


class ScriptRetriever:
    """Semantic search over the FEDE scenes and sentences Qdrant collections.

    All methods accept a pre-computed query embedding vector, keeping the
    retriever decoupled from EmbeddingGemma and from query enrichment logic.
    Those two concerns live upstream in the online pipeline.

    Typical online call order:
        1. QueryEnricher rewrites the raw user query into script style.
        2. EmbeddingGemma encodes the enriched query → query_embedding.
        3. ScriptRetriever.hierarchical_search(query_embedding, top_k) is called.
        4. Results are passed to the hybrid aggregation stage.
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        self.config = config or QdrantConfig.from_env()
        self._client = get_qdrant_client(self.config)

    # ------------------------------------------------------------------
    # Flat search methods
    # ------------------------------------------------------------------

    def search_sentences(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        movie_id_filter: Optional[str] = None,
    ) -> List[SentenceResult]:
        """Return the top-k most similar sentences to query_embedding.

        Args:
            query_embedding: Pre-computed query vector.
            top_k: Number of results to return.
            movie_id_filter: If provided, restrict results to a single movie.

        Returns:
            List of SentenceResult sorted by descending cosine similarity score.
        """
        query_filter = _build_movie_filter(movie_id_filter)
        hits = self._client.search(
            collection_name=CollectionNames.SENTENCES.value,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [_hit_to_sentence_result(h) for h in hits]

    def search_scenes(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        movie_id_filter: Optional[str] = None,
    ) -> List[SceneResult]:
        """Return the top-k most similar scenes to query_embedding.

        Args:
            query_embedding: Pre-computed query vector.
            top_k: Number of results to return.
            movie_id_filter: If provided, restrict results to a single movie.

        Returns:
            List of SceneResult sorted by descending cosine similarity score.
        """
        query_filter = _build_movie_filter(movie_id_filter)
        hits = self._client.search(
            collection_name=CollectionNames.SCENES.value,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [_hit_to_scene_result(h) for h in hits]

    # ------------------------------------------------------------------
    # Hierarchical search — primary online retrieval entry point
    # ------------------------------------------------------------------

    def hierarchical_search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        sentence_pool: int = 100,
    ) -> List[SceneResult]:
        """Two-stage hierarchical search returning ranked SceneResults.

        Stage 1 — Sentence search:
            Retrieve `sentence_pool` candidate sentences. Granular sentence
            embeddings capture specific dialogue or action details that a
            scene-level embedding might smooth over.

        Stage 2 — Scene resolution and merging:
            a. Map each retrieved sentence back to its parent scene via the
               scene_id payload field.
            b. For each parent scene, keep the highest sentence score
               (max-pooling) as its sentence-derived score.
            c. Run a direct scene search to retrieve `top_k` scene-level hits.
            d. Merge both score sets: for scenes present in both, take the
               maximum of the two scores; for scenes only in one set, use that
               score directly.
            e. Sort by merged score descending and return the top-k scenes.

        The merged ranking rewards scenes that are strongly matched by both
        granular sentence evidence and broad scene-level similarity, while
        still surfacing scenes captured by only one retrieval path.

        Args:
            query_embedding: Pre-computed query vector (same space as indexed
                embeddings — must be encoded with EmbeddingGemma).
            top_k: Number of final scene results to return.
            sentence_pool: Number of sentences to fetch in stage 1. A larger
                pool increases recall at the cost of extra scene lookups.

        Returns:
            List of SceneResult sorted by merged score, length ≤ top_k.
        """
        # Stage 1: sentence search
        sentence_hits = self.search_sentences(query_embedding, top_k=sentence_pool)

        # Build scene_id → max sentence score map
        sentence_scores: Dict[str, float] = {}
        for s in sentence_hits:
            if s.scene_id not in sentence_scores or s.score > sentence_scores[s.scene_id]:
                sentence_scores[s.scene_id] = s.score

        # Stage 2a: direct scene search
        scene_hits = self.search_scenes(query_embedding, top_k=top_k)

        # Stage 2b: index direct scene results by scene_id
        scene_results: Dict[str, SceneResult] = {h.scene_id: h for h in scene_hits}
        direct_scores: Dict[str, float] = {h.scene_id: h.score for h in scene_hits}

        # Stage 2c: merge — for scenes only reached via sentence path we need
        # to fetch their full payload from the scenes collection.
        sentence_only_ids = set(sentence_scores) - set(direct_scores)
        if sentence_only_ids:
            fetched = self._fetch_scenes_by_id(list(sentence_only_ids))
            scene_results.update(fetched)

        # Stage 2d: compute merged scores
        all_scene_ids = set(sentence_scores) | set(direct_scores)
        merged: Dict[str, float] = {}
        for sid in all_scene_ids:
            merged[sid] = max(
                sentence_scores.get(sid, 0.0),
                direct_scores.get(sid, 0.0),
            )

        # Stage 2e: sort and return top-k
        ranked_ids = sorted(merged, key=lambda sid: merged[sid], reverse=True)[:top_k]
        results = []
        for sid in ranked_ids:
            if sid in scene_results:
                result = scene_results[sid]
                result.score = merged[sid]
                results.append(result)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_scenes_by_id(self, scene_ids: List[str]) -> Dict[str, SceneResult]:
        """Fetch full scene payloads for scenes not returned by direct search.

        Uses a Qdrant payload filter to look up scenes by their scene_id
        metadata field. Score is set to 0.0 as a placeholder; the caller
        overwrites it with the merged score.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="scene_id",
                    match=MatchAny(any=scene_ids),
                )
            ]
        )
        points, _ = self._client.scroll(
            collection_name=CollectionNames.SCENES.value,
            scroll_filter=query_filter,
            limit=len(scene_ids),
            with_payload=True,
            with_vectors=False,
        )
        results: Dict[str, SceneResult] = {}
        for point in points:
            result = _point_to_scene_result(point, score=0.0)
            results[result.scene_id] = result
        return results


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------

def hierarchical_search(
    query_embedding: List[float],
    top_k: int = 20,
    sentence_pool: int = 100,
    config: Optional[QdrantConfig] = None,
) -> List[SceneResult]:
    """Module-level wrapper around ScriptRetriever.hierarchical_search."""
    return ScriptRetriever(config).hierarchical_search(
        query_embedding, top_k=top_k, sentence_pool=sentence_pool
    )


# ------------------------------------------------------------------
# Payload → result converters
# ------------------------------------------------------------------

def _build_movie_filter(movie_id: Optional[str]):
    """Return a Qdrant Filter restricting to one movie, or None."""
    if movie_id is None:
        return None

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(
        must=[FieldCondition(key="movie_id", match=MatchValue(value=movie_id))]
    )


def _hit_to_sentence_result(hit) -> SentenceResult:
    p: SentencePayload = hit.payload or {}
    return SentenceResult(
        point_id=str(hit.id),
        score=hit.score,
        movie_id=p.get("movie_id", ""),
        movie_title=p.get("movie_title", ""),
        scene_id=p.get("scene_id", ""),
        scene_index=p.get("scene_index", 0),
        text=p.get("text", ""),
        line_type=p.get("line_type", "description"),
        position_in_script=p.get("position_in_script", 0),
        character_name=p.get("character_name"),
    )


def _hit_to_scene_result(hit) -> SceneResult:
    p: ScenePayload = hit.payload or {}
    return SceneResult(
        point_id=str(hit.id),
        score=hit.score,
        movie_id=p.get("movie_id", ""),
        movie_title=p.get("movie_title", ""),
        scene_id=p.get("scene_id", ""),
        scene_index=p.get("scene_index", 0),
        text=p.get("text", ""),
        scene_title=p.get("scene_title"),
        character_names=p.get("character_names", []),
    )


def _point_to_scene_result(point, score: float) -> SceneResult:
    p: ScenePayload = point.payload or {}
    return SceneResult(
        point_id=str(point.id),
        score=score,
        movie_id=p.get("movie_id", ""),
        movie_title=p.get("movie_title", ""),
        scene_id=p.get("scene_id", ""),
        scene_index=p.get("scene_index", 0),
        text=p.get("text", ""),
        scene_title=p.get("scene_title"),
        character_names=p.get("character_names", []),
    )
