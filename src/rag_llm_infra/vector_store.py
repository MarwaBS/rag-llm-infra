"""
vector_store.py
---------------
VectorStoreProtocol: a swappable embedding-index abstraction.

The minimal surface area callers use against an embedding index. Swapping the
in-process FAISS index for a managed vector DB (Pinecone, Weaviate, Qdrant,
pgvector) is then a config change, not a rewrite.

Three implementations ship:

  - `FAISSVectorStore`   : in-process FAISS IndexFlatIP (what `auto` picks
                           when FAISS imports)
  - `NumpyVectorStore`   : pure-numpy, and what `auto` falls back to when
                           FAISS is missing or will not load
  - `QdrantVectorStore`  : real, tested Qdrant backend. Defaults to
                           `QdrantClient(":memory:")` for test parity;
                           set `QDRANT_URL` to point at a managed Qdrant
                           instance in production. A managed backend behind
                           the same Protocol, so the abstraction is
                           exercised rather than asserted.

The `get_vector_store()` factory selects an implementation by name.
Default is "auto" -> FAISS when available, NumPy otherwise.

That preference is measured, not assumed. `benchmarks/backend_crossover.py`
(384 dims, k=5, brute-force inner product) puts FAISS ahead at every size it
tries, from a hundred documents to a hundred thousand. So `auto` is right
wherever it is asked, and below a thousand documents either backend answers in
under half a millisecond. The margin itself moves too much between runs on one
host to quote, so run the script for it. Both backends return the same rows;
only the tie order differs, which `search` documents.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

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

# Type aliases for the float32/int64 arrays we operate on. PEP 695 `type`
# statement (py3.12) so mypy treats them as types, not module-level variables
# (a plain assignment fails with `[valid-type]` under the CI mypy run).
type NDArrayF32 = npt.NDArray[np.float32]
type NDArrayI64 = npt.NDArray[np.int64]


def _as_2d_float32(arr: Any, name: str, *, copy: bool = False) -> NDArrayF32:
    """Validate and coerce an embedding/query batch to an `(N, D)` float32 array.

    Raises `ValueError` for a non-2-D input or for non-finite values. Otherwise a
    1-D `add` surfaces as an opaque `AxisError` and a NaN reaches the backend as
    silent garbage scores. Pass `copy=True` when the caller normalizes in place
    (FAISS) so the caller's own array is never mutated.
    """
    a = np.array(arr, dtype=np.float32) if copy else np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D (N, D) float array; got {a.ndim}-D with shape "
            f"{a.shape}. Reshape a single vector via arr.reshape(1, -1)."
        )
    if a.size and not np.isfinite(a).all():
        raise ValueError(
            f"{name} contains non-finite values (NaN/inf); embeddings must be finite."
        )
    return a


def _empty_result(n_queries: int) -> tuple[NDArrayF32, NDArrayI64]:
    """An empty store (`size == 0`) returns width `min(k, size) == 0` on every
    backend. The contract says the case is legal, so no backend raises its own
    error for it."""
    return (
        np.empty((n_queries, 0), dtype=np.float32),
        np.empty((n_queries, 0), dtype=np.int64),
    )


# One source of truth for "is FAISS importable on this host".
try:
    from .evidence_index import FAISS_AVAILABLE as _FAISS_AVAILABLE

    FAISS_AVAILABLE: bool = _FAISS_AVAILABLE
except ImportError:
    logger.debug("faiss capability flag unreadable; assuming unavailable")
    FAISS_AVAILABLE = False
except Exception as exc:
    # A ctypes load inside the package raises OSError, not ImportError.
    logger.warning("FAISS is installed but failed to load: %s", exc)
    FAISS_AVAILABLE = False

if FAISS_AVAILABLE:
    import faiss

# Qdrant is an optional dev/ops dependency. Import lazily so the module
# loads cleanly in environments that don't install it.
try:
    from qdrant_client import QdrantClient
    from qdrant_client import models as qdrant_models

    QDRANT_AVAILABLE: bool = True
except ImportError:
    logger.debug("qdrant-client not installed; the qdrant backend is unavailable")
    QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore[misc,assignment]
    qdrant_models = None  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - runs only in the subprocess
    # test_broken_optional_deps spawns, which in-process coverage cannot see.
    logger.warning("qdrant-client is installed but failed to load: %s", exc)
    QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore[misc,assignment]
    qdrant_models = None  # type: ignore[assignment]


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Minimal vector store contract for embedding retrieval.

    Implementations must accept already-computed `(N, D)` float32 embeddings
    and return per-query top-k similarity scores. Normalization and any
    backend-specific indexing live behind the implementation. Callers do
    not normalize before calling `add` or `search`.
    """

    backend_name: str
    backend_version: str

    def add(self, embeddings: NDArrayF32) -> None:
        """Build/replace the index from `(N, D)` float32 embeddings."""
        ...

    def search(self, queries: NDArrayF32, k: int) -> tuple[NDArrayF32, NDArrayI64]:
        """Return `(distances, indices)` arrays of shape `(Nq, min(k, size))`.

        Ties: each backend orders equal scores its own way, and they do not
        agree. NumPy breaks them on the lower document index, FAISS and Qdrant
        do not. Swapping backends can therefore reorder equally-scoring
        documents. What every backend does guarantee is repeatability: the same
        store and query return the same rows in the same order, in this process
        and the next. Where more documents share the boundary score than there
        are slots left, which of them appear is unspecified.

        A store cannot return more results than it holds. When `k > size` every
        backend truncates to `size` rather than padding, so the row width is
        `min(k, size)` across FAISS, NumPy and Qdrant alike. An empty
        store (`size == 0`, e.g. built from a zero-row `add`) therefore returns
        `(Nq, 0)`-shaped arrays uniformly, not a backend-specific error. Calling
        `search` before any `add` is a different case, a programming error, and
        raises `RuntimeError`. `k` must be >= 1. Distances are inner-product
        similarities in `[-1, 1]`, equal to cosine because both sides are
        L2-normalized.
        """
        ...

    @property
    def size(self) -> int:
        """Number of vectors currently in the store."""
        ...

    @property
    def is_native(self) -> bool:
        """True if backed by a native vector index (e.g. FAISS), False for a
        plain numpy matrix. A caller that persists an index needs to know
        whether the bytes depend on a native library's version."""
        ...

    def reset(self) -> None:
        """Drop all vectors and free backend resources."""
        ...


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
        self._index: Any | None = None
        self.backend_version = faiss.__version__

    def add(self, embeddings: NDArrayF32) -> None:
        # copy=True: faiss.normalize_L2 normalizes in place, so without a copy a
        # float32 caller would have its own array silently normalized as a side
        # effect (NumPy/Qdrant copy, so the backends would disagree).
        embeddings = _as_2d_float32(embeddings, "embeddings", copy=True)
        dim = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self._index = index

    def search(self, queries: NDArrayF32, k: int) -> tuple[NDArrayF32, NDArrayI64]:
        if self._index is None:
            raise RuntimeError("FAISSVectorStore.search() called before add()")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        queries = _as_2d_float32(queries, "queries", copy=True)
        if queries.shape[1] != self._index.d:
            raise ValueError(
                f"query dim {queries.shape[1]} != index dim {self._index.d}"
            )
        if self.size == 0:
            return _empty_result(queries.shape[0])
        faiss.normalize_L2(queries)
        # Ask FAISS for at most `size` neighbours so the row width is
        # min(k, size), matching NumPy/Qdrant, instead of FAISS's -1/-inf
        # padding when k > size.
        k_eff = min(k, self.size)
        scores, indices = self._index.search(queries, k_eff)
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
            except Exception as exc:
                logger.warning("FAISS reset failed: %s", exc)
        self._index = None


class NumpyVectorStore:
    """Pure-numpy fallback. Stores `(N, D)` row-normalized matrix and runs
    cosine similarity via a single matmul. Every query scores every vector,
    so cost is linear in corpus size. FAISS was ahead at every size
    `benchmarks/backend_crossover.py` measures, so this is the fallback rather
    than a choice; measure your own corpus before relying on either.
    """

    backend_name = "numpy"
    backend_version = np.__version__

    def __init__(self) -> None:
        self._matrix: NDArrayF32 | None = None

    def add(self, embeddings: NDArrayF32) -> None:
        embeddings = _as_2d_float32(embeddings, "embeddings")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = embeddings / norms

    def search(self, queries: NDArrayF32, k: int) -> tuple[NDArrayF32, NDArrayI64]:
        if self._matrix is None:
            raise RuntimeError("NumpyVectorStore.search() called before add()")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        queries = _as_2d_float32(queries, "queries")
        if queries.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"query dim {queries.shape[1]} != index dim {self._matrix.shape[1]}"
            )
        if self.size == 0:
            return _empty_result(queries.shape[0])
        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        norm_queries = queries / q_norms
        # (N, D) @ (D, Nq) -> (N, Nq), then transpose to (Nq, N)
        similarities = np.dot(self._matrix, norm_queries.T).T  # (Nq, N)
        k_eff = min(k, similarities.shape[1])
        # argpartition for top-k, then sort descending within the top slice
        top_idx = np.argpartition(-similarities, k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(similarities.shape[0])[:, None]
        top_scores = similarities[rows, top_idx]
        # lexsort's last key is the primary one: score descending, then the
        # lower document index. `argsort` defaults to quicksort, which is not
        # stable, so equal scores would come back in whatever order the
        # partition happened to leave them in.
        order = np.lexsort((top_idx, -top_scores), axis=1)
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


class QdrantVectorStore:
    """Real Qdrant backend against `qdrant-client`.

    **This store owns `collection` exclusively.** `add()` replaces the index, so
    it deletes and recreates that collection. Point two stores at one name and
    the second `add()` destroys the first one's vectors. The name is therefore
    required and has no default: a shared endpoint plus a default name is a
    collision nobody chose.

    **This backend can return `-1` index and `-1.0` score.** `search` answers a
    batch as one rectangular array. A row with fewer hits than asked for is
    therefore padded, not truncated. That happens when the collection changes
    between the count and the query. Filter on `index >= 0`. FAISS and NumPy
    never pad.

    Defaults to `QdrantClient(":memory:")`, a full in-process Qdrant, the same
    code path as a managed one, with no server to run for tests. Set `QDRANT_URL`
    to point at a managed endpoint; it is read on construction.

    A third backend behind the same Protocol, so the swap path is executed
    rather than merely described.
    """

    backend_name = "qdrant"

    def __init__(self, collection: str, url: str | None = None) -> None:
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
        self._dim: int | None = None
        # __version__ is a module attribute, not a class attribute.
        import qdrant_client as _qc

        self.backend_version = getattr(_qc, "__version__", "unknown")

    def _ensure_collection(self, dim: int) -> None:
        """Create or recreate the owned collection with cosine distance.

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
        embeddings = _as_2d_float32(embeddings, "embeddings")
        dim = int(embeddings.shape[1])
        # Qdrant normalizes internally when distance=COSINE, but we still
        # L2-normalize here so the `is_native=True` contract (scores in
        # [-1, 1]) matches FAISS/NumPy backends exactly.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms
        self._ensure_collection(dim)
        # Skip the upsert for a zero-row add. An empty points list trips some
        # qdrant-client versions. `_dim` is still set, so search distinguishes
        # "empty store" (returns (Nq, 0)) from "before add" (raises).
        if normed.shape[0]:
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

    def search(self, queries: NDArrayF32, k: int) -> tuple[NDArrayF32, NDArrayI64]:
        if self._dim is None:
            raise RuntimeError("QdrantVectorStore.search() called before add()")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        queries = _as_2d_float32(queries, "queries")
        if queries.shape[1] != self._dim:
            raise ValueError(f"query dim {queries.shape[1]} != index dim {self._dim}")
        held = self.size
        if held == 0:
            return _empty_result(queries.shape[0])
        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        normed_queries = queries / q_norms
        k_eff = min(k, held)

        # Batched search: a single HTTP round-trip for every query at once.
        # Looping `query_points` once per query adds up to dozens of
        # round-trips when many queries run together; `query_batch_points`
        # sends them as one request.
        # Falls back to the per-query loop only if the client version does
        # not expose the batched method (qdrant-client < 1.8).
        batch_fn = getattr(self._client, "query_batch_points", None)
        if batch_fn is not None:
            requests = [
                qdrant_models.QueryRequest(
                    query=q.tolist(), limit=k_eff, with_payload=False
                )
                for q in normed_queries
            ]
            responses = batch_fn(collection_name=self._collection, requests=requests)
            hit_lists = [resp.points for resp in responses]
        else:
            hit_lists = []
            for q in normed_queries:
                resp = self._client.query_points(
                    collection_name=self._collection,
                    query=q.tolist(),
                    limit=k_eff,
                )
                hit_lists.append(resp.points)

        scores_list = []
        indices_list = []
        for hits in hit_lists:
            row_scores = [float(h.score) for h in hits]
            row_indices = [int(h.id) for h in hits]
            # A batch answers as one rectangular array, so a short row is padded
            # rather than the whole batch truncated to the shortest. Index -1
            # marks the padding; see the class docstring.
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
        """Counted in the collection, not remembered locally.

        A local count goes stale the moment anything else writes, and `search`
        derives its row width from this, so a stale number pads the answer with
        sentinel indices. Costs one round-trip.

        Raises `ValueError` from the client if the collection has been dropped
        since `add()`. `search` inherits that, which is why the class docstring
        says this store owns its collection.
        """
        if self._dim is None:
            return 0
        return int(self._client.count(collection_name=self._collection).count)

    @property
    def is_native(self) -> bool:
        return True

    def reset(self) -> None:
        if self._dim is not None:
            try:
                self._client.delete_collection(collection_name=self._collection)
            except Exception as exc:
                logger.warning("Qdrant reset failed: %s", exc)
        self._dim = None


def get_vector_store(
    backend: str = "auto", *, collection: str | None = None
) -> VectorStoreProtocol:
    """Return a configured `VectorStoreProtocol` instance.

    `backend` values:
      - "auto"      -> FAISS when available, NumPy otherwise (default)
      - "faiss"     -> FAISS, error if not installed
      - "numpy"     -> NumPy fallback (always available)
      - "qdrant"    -> real Qdrant backend via qdrant-client (embedded or
                      managed, depending on QDRANT_URL env). Needs `collection`,
                      which that store replaces on every `add()` and therefore
                      owns.
    """
    backend_normalized = (backend or "auto").lower().strip()
    if collection is not None and backend_normalized != "qdrant":
        raise ValueError(f"collection= applies to qdrant, not {backend_normalized!r}")
    if backend_normalized == "auto":
        if FAISS_AVAILABLE:
            return FAISSVectorStore()
        return NumpyVectorStore()
    if backend_normalized == "faiss":
        return FAISSVectorStore()
    if backend_normalized == "numpy":
        return NumpyVectorStore()
    if backend_normalized == "qdrant":
        if collection is None:
            raise ValueError(
                "qdrant needs collection=: add() replaces that collection's "
                "contents, so the store must own the name it is given."
            )
        return QdrantVectorStore(collection=collection)
    raise ValueError(
        f"Unknown vector_store_backend={backend!r}. "
        "Valid: auto | faiss | numpy | qdrant"
    )
