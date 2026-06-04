"""
vector_store.py
---------------
VectorStoreProtocol — a swappable embedding-index abstraction.

This module defines the minimal surface area callers use against an
embedding index, so that swapping the in-process FAISS index
for a managed vector DB (Pinecone, Weaviate, Qdrant, pgvector) is a
config change rather than a rewrite.

Three implementations ship:

  - `FAISSVectorStore`   — in-process FAISS IndexFlatIP (default when
                           FAISS is available)
  - `NumpyVectorStore`   — pure-numpy fallback for environments without
                           a working FAISS install (also used by the
                           engine for small corpora, `len(chunks) <= 10`)
  - `QdrantVectorStore`  — real, tested Qdrant backend. Defaults to
                           `QdrantClient(":memory:")` for test parity;
                           set `QDRANT_URL` to point at a managed Qdrant
                           instance in production. This replaces the old
                           Pinecone stub and proves the Protocol
                           abstraction is swap-able end-to-end, not
                           aspirational.

The `get_vector_store()` factory selects an implementation by name.
Default is "auto" → FAISS when available, NumPy otherwise.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, Tuple, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

__all__ = [
    "VectorStoreProtocol",
    "FAISSVectorStore",
    "NumpyVectorStore",
    "QdrantVectorStore",
    "get_vector_store",
    "FAISS_AVAILABLE",
    "QDRANT_AVAILABLE",
]

# Type aliases for the float32/int64 arrays we operate on. Must use the
# `TypeAlias` annotation (PEP 613) so mypy treats them as types, not as
# module-level variables (the latter fails with `[valid-type]` under the
# broader CI lint mypy run).
NDArrayF32: TypeAlias = npt.NDArray[np.float32]
NDArrayI64: TypeAlias = npt.NDArray[np.int64]

# Re-use the existing capability flag from evidence_index so we have one
# source of truth for "is FAISS importable on this host".
try:
    from evidence_index import FAISS_AVAILABLE as _FAISS_AVAILABLE
    FAISS_AVAILABLE: bool = _FAISS_AVAILABLE
except Exception:  # pragma: no cover - defensive
    FAISS_AVAILABLE = False

if FAISS_AVAILABLE:
    import faiss

# Qdrant is an optional dev/ops dependency — import lazily so the module
# loads cleanly in environments that don't install it.
try:
    from qdrant_client import QdrantClient, models as qdrant_models
    QDRANT_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised in envs without qdrant-client
    QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore[misc,assignment]
    qdrant_models = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Protocol — the surface area callers depend on
# ---------------------------------------------------------------------------
@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Minimal vector store contract for embedding retrieval.

    Implementations must accept already-computed `(N, D)` float32 embeddings
    and return per-query top-k similarity scores. Normalization and any
    backend-specific indexing live behind the implementation — callers do
    not normalize before calling `add` or `search`.
    """

    backend_name: str
    backend_version: str

    def add(self, embeddings: NDArrayF32) -> None:
        """Build/replace the index from `(N, D)` float32 embeddings."""
        ...

    def search(self, queries: NDArrayF32, k: int) -> Tuple[NDArrayF32, NDArrayI64]:
        """Return `(distances, indices)` arrays of shape `(Nq, k)`.

        Distances are inner-product similarities in `[-1, 1]` (the engine
        treats them as cosine because both sides are L2-normalized).
        """
        ...

    @property
    def size(self) -> int:
        """Number of vectors currently in the store."""
        ...

    @property
    def is_native(self) -> bool:
        """True if backed by a native vector index (e.g., FAISS), False if
        backed by a plain numpy matrix. Used by the cache layer to decide
        whether the version stamp must match across reloads."""
        ...

    def reset(self) -> None:
        """Drop all vectors and free backend resources."""
        ...


# ---------------------------------------------------------------------------
# FAISS implementation — the production default when FAISS is installed
# ---------------------------------------------------------------------------
class FAISSVectorStore:
    """In-process FAISS `IndexFlatIP` over L2-normalized embeddings.

    Holds the index behind the `VectorStoreProtocol` surface instead of an
    opaque `faiss.Index`, so callers depend on the protocol, not FAISS.
    """

    backend_name = "faiss"

    def __init__(self) -> None:
        if not FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISSVectorStore requires the `faiss` package. "
                "Install `faiss-cpu` or set vector_store_backend=numpy."
            )
        self._index: Optional[Any] = None
        self.backend_version = faiss.__version__

    def add(self, embeddings: NDArrayF32) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self._index = index

    def search(self, queries: NDArrayF32, k: int) -> Tuple[NDArrayF32, NDArrayI64]:
        if self._index is None:
            raise RuntimeError("FAISSVectorStore.search() called before add()")
        if queries.dtype != np.float32:
            queries = queries.astype("float32")
        faiss.normalize_L2(queries)
        scores, indices = self._index.search(queries, k)
        return scores, indices

    @property
    def size(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    @property
    def is_native(self) -> bool:
        return True

    def reset(self) -> None:
        if self._index is not None and hasattr(self._index, "reset"):
            try:
                self._index.reset()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("FAISS reset failed: %s", exc)
        self._index = None


# ---------------------------------------------------------------------------
# Numpy fallback — used when FAISS is not installed or the corpus is tiny
# ---------------------------------------------------------------------------
class NumpyVectorStore:
    """Pure-numpy fallback. Stores `(N, D)` row-normalized matrix and runs
    cosine similarity via a single matmul. Acceptable for small corpora
    (the engine's existing rule: `len(chunks) <= 10` → numpy).
    """

    backend_name = "numpy"
    backend_version = np.__version__

    def __init__(self) -> None:
        self._matrix: Optional[NDArrayF32] = None

    def add(self, embeddings: NDArrayF32) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = embeddings / norms

    def search(self, queries: NDArrayF32, k: int) -> Tuple[NDArrayF32, NDArrayI64]:
        if self._matrix is None:
            raise RuntimeError("NumpyVectorStore.search() called before add()")
        if queries.dtype != np.float32:
            queries = queries.astype("float32")
        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        norm_queries = queries / q_norms
        # (N, D) @ (D, Nq) → (N, Nq), then transpose to (Nq, N)
        similarities = np.dot(self._matrix, norm_queries.T).T  # (Nq, N)
        k_eff = min(k, similarities.shape[1])
        # argpartition for top-k, then sort descending within the top slice
        top_idx = np.argpartition(-similarities, k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(similarities.shape[0])[:, None]
        top_scores = similarities[rows, top_idx]
        # Sort each row descending by score
        order = np.argsort(-top_scores, axis=1)
        sorted_scores = np.take_along_axis(top_scores, order, axis=1)
        sorted_indices = np.take_along_axis(top_idx, order, axis=1)
        return sorted_scores.astype("float32"), sorted_indices.astype("int64")

    @property
    def size(self) -> int:
        return int(self._matrix.shape[0]) if self._matrix is not None else 0

    @property
    def is_native(self) -> bool:
        return False

    def reset(self) -> None:
        self._matrix = None


# ---------------------------------------------------------------------------
# Qdrant implementation — proves the abstraction's swap-ability end-to-end
# ---------------------------------------------------------------------------
class QdrantVectorStore:
    """Real Qdrant backend against `qdrant-client`.

    Defaults to `QdrantClient(":memory:")` which runs a full in-process
    Qdrant instance — same code path as a managed Qdrant, no external
    server needed for tests. Production callers can point at a managed
    endpoint by setting the `QDRANT_URL` environment variable (read on
    construction).

    Replaces the old `PineconeVectorStore` stub. The earlier stub
    documented the swap path; this class executes it.
    """

    backend_name = "qdrant"

    def __init__(self, url: Optional[str] = None, collection: str = "evidence") -> None:
        if not QDRANT_AVAILABLE:
            raise RuntimeError(
                "QdrantVectorStore requires `qdrant-client`. "
                "Install with `pip install qdrant-client` or pick a different "
                "vector_store_backend (auto|faiss|numpy)."
            )
        import os as _os
        self._url = url or _os.getenv("QDRANT_URL") or ":memory:"
        self._collection = collection
        # `QdrantClient(":memory:")` is the embedded in-process mode;
        # `QdrantClient(url="http://...")` points at a managed instance.
        if self._url == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=self._url)
        self._dim: Optional[int] = None
        self._size: int = 0
        self.backend_version = getattr(QdrantClient, "__version__", "unknown")

    def _ensure_collection(self, dim: int) -> None:
        """Create or recreate the collection with cosine distance.

        Uses `delete_collection` + `create_collection` instead of the
        deprecated `recreate_collection` (qdrant-client >= 1.12).
        """
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(collection_name=self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qdrant_models.VectorParams(
                size=dim,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        self._dim = dim

    def add(self, embeddings: NDArrayF32) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        dim = int(embeddings.shape[1])
        # Qdrant normalizes internally when distance=COSINE, but we still
        # L2-normalize here so the `is_native=True` contract (scores in
        # [-1, 1]) matches FAISS/NumPy backends exactly.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms
        self._ensure_collection(dim)
        self._client.upsert(
            collection_name=self._collection,
            points=[
                qdrant_models.PointStruct(
                    id=int(i),
                    vector=normed[i].tolist(),
                )
                for i in range(normed.shape[0])
            ],
        )
        self._size = int(normed.shape[0])

    def search(self, queries: NDArrayF32, k: int) -> Tuple[NDArrayF32, NDArrayI64]:
        if self._dim is None or self._size == 0:
            raise RuntimeError("QdrantVectorStore.search() called before add()")
        if queries.dtype != np.float32:
            queries = queries.astype("float32")
        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        normed_queries = queries / q_norms
        k_eff = min(k, self._size)

        # Batched search — a single HTTP round-trip for every query at once.
        # Looping `query_points` once per query adds up to dozens of
        # round-trips when many queries run together; `query_batch_points`
        # sends them as one request.
        # Falls back to the per-query loop only if the client version does
        # not expose the batched method (qdrant-client < 1.8).
        batch_fn = getattr(self._client, "query_batch_points", None)
        if batch_fn is not None:
            requests = [
                qdrant_models.QueryRequest(query=q.tolist(), limit=k_eff, with_payload=False)
                for q in normed_queries
            ]
            responses = batch_fn(collection_name=self._collection, requests=requests)
            hit_lists = [resp.points for resp in responses]
        else:
            hit_lists = []
            for q in normed_queries:
                resp = self._client.query_points(
                    collection_name=self._collection, query=q.tolist(), limit=k_eff,
                )
                hit_lists.append(resp.points)

        scores_list = []
        indices_list = []
        for hits in hit_lists:
            row_scores = [float(h.score) for h in hits]
            row_indices = [int(h.id) for h in hits]
            # Pad to k_eff with sentinel values if Qdrant returned fewer hits.
            while len(row_scores) < k_eff:
                row_scores.append(-1.0)
                row_indices.append(-1)
            scores_list.append(row_scores)
            indices_list.append(row_indices)
        scores = np.asarray(scores_list, dtype="float32")
        indices = np.asarray(indices_list, dtype="int64")
        return scores, indices

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_native(self) -> bool:
        return True

    def reset(self) -> None:
        if self._dim is not None:
            try:
                self._client.delete_collection(collection_name=self._collection)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Qdrant reset failed: %s", exc)
        self._dim = None
        self._size = 0


# ---------------------------------------------------------------------------
# Factory — env-selected implementation
# ---------------------------------------------------------------------------
def get_vector_store(backend: str = "auto") -> VectorStoreProtocol:
    """Return a configured `VectorStoreProtocol` instance.

    `backend` values:
      - "auto"      → FAISS when available, NumPy otherwise (default)
      - "faiss"     → FAISS, error if not installed
      - "numpy"     → NumPy fallback (always available)
      - "qdrant"    → real Qdrant backend via qdrant-client (embedded or
                      managed, depending on QDRANT_URL env)
    """
    backend_normalized = (backend or "auto").lower().strip()
    if backend_normalized == "auto":
        if FAISS_AVAILABLE:
            return FAISSVectorStore()
        return NumpyVectorStore()
    if backend_normalized == "faiss":
        return FAISSVectorStore()
    if backend_normalized == "numpy":
        return NumpyVectorStore()
    if backend_normalized == "qdrant":
        return QdrantVectorStore()
    raise ValueError(
        f"Unknown vector_store_backend={backend!r}. "
        "Valid: auto | faiss | numpy | qdrant"
    )
