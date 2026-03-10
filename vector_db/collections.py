"""Collection management for FEDE's Qdrant collections (scenes and sentences)."""

from typing import Any, Dict, List, Optional

from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
    VectorParamsDiff,
)

from .client import get_qdrant_client
from .config import QdrantConfig
from .schemas import CollectionNames

_COLLECTION_NAMES: List[CollectionNames] = [
    CollectionNames.SCENES,
    CollectionNames.SENTENCES,
]


class CollectionManager:
    """Creates and manages the scenes and sentences Qdrant collections.

    Both collections share the same vector parameters (cosine distance,
    configurable HNSW and INT8 quantization) derived from QdrantConfig.
    """

    def __init__(
        self,
        config: Optional[QdrantConfig] = None,
        vector_size: Optional[int] = None,
    ):
        """Initialise the collection manager.

        Args:
            config: Qdrant configuration. If None, loads from environment.
            vector_size: Override for vector dimensionality. Defaults to
                config.vector_size (768 for EmbeddingGemma).
        """
        self.config = config or QdrantConfig.from_env()
        resolved_vector_size = (
            self.config.vector_size if vector_size is None else vector_size
        )

        if resolved_vector_size <= 0:
            raise ValueError("vector_size must be > 0")

        self.client = get_qdrant_client(self.config)
        self.vector_size = resolved_vector_size
        self._collections: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_collections(self) -> None:
        """Create scenes and sentences collections if they do not exist."""
        for name in _COLLECTION_NAMES:
            self._create_collection_if_missing(name.value)

    def get_collection(self, collection_name: CollectionNames) -> Any:
        """Return cached collection metadata, fetching from Qdrant if needed."""
        if collection_name.value in self._collections:
            return self._collections[collection_name.value]

        if not self.client.collection_exists(collection_name=collection_name.value):
            raise ValueError(
                f"Collection '{collection_name.value}' does not exist. "
                "Call initialize_collections() first."
            )

        collection = self.client.get_collection(collection_name=collection_name.value)
        self._validate_collection_schema(collection_name.value, collection)
        self._collections[collection_name.value] = collection
        return collection

    def reset_collection(self, collection_name: CollectionNames) -> None:
        """Delete and recreate a collection, wiping all indexed vectors."""
        try:
            self.client.delete_collection(collection_name=collection_name.value)
        except Exception:
            pass

        self._collections.pop(collection_name.value, None)
        self._create_collection_if_missing(collection_name.value)

    def apply_tuning(self, collection_name: CollectionNames) -> None:
        """Push current config's storage/HNSW/quantization settings to an
        existing collection without recreating it.

        Useful when config changes (e.g. toggling on-disk) need to be applied
        to a collection that was created under different settings.
        """
        self.client.update_collection(
            collection_name=collection_name.value,
            vectors_config={
                "": VectorParamsDiff(on_disk=self.config.scripts_on_disk),
            },
            hnsw_config=HnswConfigDiff(
                on_disk=self.config.hnsw_on_disk,
                m=self.config.hnsw_m,
            ),
            quantization_config=self._quantization_config(),
        )

    def list_collections(self) -> List[str]:
        """Return names of all collections currently in Qdrant."""
        response = self.client.get_collections()
        return [c.name for c in response.collections]

    def collection_exists(self, collection_name: CollectionNames) -> bool:
        """Return True if the collection exists in Qdrant."""
        try:
            return self.client.collection_exists(collection_name=collection_name.value)
        except Exception:
            return False

    def get_collection_count(self, collection_name: CollectionNames) -> int:
        """Return the exact number of vectors indexed in a collection."""
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name.value}' does not exist")

        response = self.client.count(
            collection_name=collection_name.value,
            exact=True,
        )
        return int(response.count)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_collection_if_missing(self, name: str) -> Any:
        """Create a collection if absent, then cache and return its metadata."""
        if not self.client.collection_exists(collection_name=name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=self._build_vector_params(),
            )

        collection = self.client.get_collection(collection_name=name)
        self._validate_collection_schema(name, collection)
        self._collections[name] = collection
        return collection

    def _build_vector_params(self) -> VectorParams:
        """Build VectorParams from current config, applied identically to both
        collections."""
        params = VectorParams(
            size=self.vector_size,
            distance=Distance.COSINE,
            on_disk=self.config.scripts_on_disk,
            hnsw_config=HnswConfigDiff(
                on_disk=self.config.hnsw_on_disk,
                m=self.config.hnsw_m,
            ),
        )
        quantization = self._quantization_config()
        if quantization is not None:
            params.quantization_config = quantization
        return params

    def _quantization_config(self) -> Optional[ScalarQuantization]:
        """Return INT8 scalar quantization config, or None if disabled."""
        if not self.config.int8_quantization:
            return None

        return ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=self.config.quantile,
                always_ram=self.config.quantization_always_ram,
            )
        )

    def _validate_collection_schema(self, collection_name: str, collection: Any) -> None:
        """Raise if the existing collection's vector size or distance metric
        does not match the current config."""
        vector_params = self._extract_vector_params(collection)
        if vector_params is None:
            return

        size = getattr(vector_params, "size", None)
        if isinstance(size, int) and size != self.vector_size:
            raise ValueError(
                f"Collection '{collection_name}' has vector size {size}, "
                f"but QDRANT_VECTOR_SIZE is {self.vector_size}. "
                "Reset or recreate the collection with matching dimensions."
            )

        distance = getattr(vector_params, "distance", None)
        if isinstance(distance, Distance) and distance != Distance.COSINE:
            raise ValueError(
                f"Collection '{collection_name}' uses distance {distance}, "
                "but FEDE expects COSINE."
            )

    @staticmethod
    def _extract_vector_params(collection: Any) -> Optional[Any]:
        config = getattr(collection, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            return None

        if isinstance(vectors, dict):
            return vectors.get("") or next(iter(vectors.values()), None)

        return vectors


# ------------------------------------------------------------------
# Module-level convenience functions
# ------------------------------------------------------------------

def initialize_all_collections(config: Optional[QdrantConfig] = None) -> CollectionManager:
    """Create scenes and sentences collections and return the manager."""
    manager = CollectionManager(config)
    manager.initialize_collections()
    return manager


def get_scenes_collection(config: Optional[QdrantConfig] = None) -> Any:
    """Return metadata for the scenes collection."""
    manager = CollectionManager(config)
    return manager.get_collection(CollectionNames.SCENES)


def get_sentences_collection(config: Optional[QdrantConfig] = None) -> Any:
    """Return metadata for the sentences collection."""
    manager = CollectionManager(config)
    return manager.get_collection(CollectionNames.SENTENCES)
