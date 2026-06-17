"""VectorStoreProtocol conformance + known-answer search across backends.

NumPy is always exercised; FAISS and Qdrant tests skip cleanly when those
optional backends are not installed.
"""

import numpy as np
import pytest

from rag_llm_infra.vector_store import (
    FAISS_AVAILABLE,
    QDRANT_AVAILABLE,
    FAISSVectorStore,
    NumpyVectorStore,
    QdrantVectorStore,
    VectorStoreProtocol,
    get_vector_store,
)


def _orthonormal_corpus() -> np.ndarray:
    # Three orthonormal vectors → the nearest-neighbour answer is unambiguous.
    return np.eye(3, dtype="float32")


class TestNumpyBackend:
    def test_conforms_to_protocol(self) -> None:
        assert isinstance(NumpyVectorStore(), VectorStoreProtocol)

    def test_known_answer_search(self) -> None:
        store = NumpyVectorStore()
        store.add(_orthonormal_corpus())
        scores, idx = store.search(np.array([[0, 1, 0]], dtype="float32"), k=1)
        assert idx[0][0] == 1
        assert scores[0][0] == pytest.approx(1.0, abs=1e-5)

    def test_size_and_reset(self) -> None:
        store = NumpyVectorStore()
        store.add(_orthonormal_corpus())
        assert store.size == 3
        assert store.is_native is False
        store.reset()
        assert store.size == 0

    def test_search_before_add_raises(self) -> None:
        with pytest.raises(RuntimeError):
            NumpyVectorStore().search(np.eye(1, dtype="float32"), k=1)


class TestFactory:
    def test_routes_numpy(self) -> None:
        assert get_vector_store("numpy").backend_name == "numpy"

    def test_auto_is_available(self) -> None:
        assert isinstance(get_vector_store("auto"), VectorStoreProtocol)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown vector_store_backend"):
            get_vector_store("pinecone")


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
class TestFaissBackend:
    def test_conforms_and_known_answer(self) -> None:
        store = FAISSVectorStore()
        assert isinstance(store, VectorStoreProtocol)
        store.add(_orthonormal_corpus())
        _, idx = store.search(np.array([[1, 0, 0]], dtype="float32"), k=1)
        assert idx[0][0] == 0
        assert store.is_native is True


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantBackend:
    def test_conforms_and_known_answer(self) -> None:
        store = QdrantVectorStore()
        assert isinstance(store, VectorStoreProtocol)
        store.add(_orthonormal_corpus())
        _, idx = store.search(np.array([[0, 0, 1]], dtype="float32"), k=1)
        assert idx[0][0] == 2


def _available_backends() -> list:
    items = [("numpy", NumpyVectorStore)]
    if FAISS_AVAILABLE:
        items.append(("faiss", FAISSVectorStore))
    if QDRANT_AVAILABLE:
        items.append(("qdrant", QdrantVectorStore))
    return items


@pytest.mark.parametrize("name,cls", _available_backends())
class TestContractAcrossBackends:
    """Behaviours that MUST hold identically on every backend (the protocol
    contract), so a swap is truly transparent."""

    def test_k_greater_than_size_truncates_to_size(self, name, cls) -> None:
        # Row width is min(k, size) on every backend — no FAISS -1/-inf padding.
        store = cls()
        store.add(_orthonormal_corpus())  # size == 3
        scores, idx = store.search(np.array([[1, 0, 0]], dtype="float32"), k=10)
        assert idx.shape == (1, 3), f"{name}: expected (1, 3), got {idx.shape}"
        assert scores.shape == (1, 3), f"{name}: scores shape {scores.shape}"

    def test_k_below_one_raises(self, name, cls) -> None:
        store = cls()
        store.add(_orthonormal_corpus())
        with pytest.raises(ValueError):
            store.search(np.array([[1, 0, 0]], dtype="float32"), k=0)

    def test_add_does_not_mutate_caller_array(self, name, cls) -> None:
        # FAISS normalized the caller's float32 array in place; NumPy/Qdrant
        # copied. Use non-unit vectors so any in-place normalize is detectable.
        store = cls()
        corpus = np.array(
            [[3.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]], dtype="float32"
        )
        before = corpus.copy()
        store.add(corpus)
        assert np.array_equal(corpus, before), (
            f"{name} mutated the caller's array in add()"
        )

    def test_search_does_not_mutate_query_array(self, name, cls) -> None:
        store = cls()
        store.add(_orthonormal_corpus())
        q = np.array([[3.0, 4.0, 0.0]], dtype="float32")
        before = q.copy()
        store.search(q, k=1)
        assert np.array_equal(q, before), f"{name} mutated the query array in search()"

    def test_empty_store_returns_zero_width(self, name, cls) -> None:
        # Regression: an empty store (built from a 0-row add) used to diverge —
        # FAISS raised a bare AssertionError, Qdrant a misleading "called before
        # add()". The documented contract is row width min(k, size) == 0, which
        # NumPy already honoured; now all three return (Nq, 0) uniformly.
        store = cls()
        store.add(np.zeros((0, 3), dtype="float32"))
        assert store.size == 0
        scores, idx = store.search(np.eye(2, 3, dtype="float32"), k=5)
        assert scores.shape == (2, 0), f"{name}: scores {scores.shape}, want (2, 0)"
        assert idx.shape == (2, 0), f"{name}: idx {idx.shape}, want (2, 0)"

    def test_search_before_add_raises_runtimeerror(self, name, cls) -> None:
        # Distinct from the empty-store case above: never calling add() is a
        # programming error and must raise RuntimeError on every backend.
        with pytest.raises(RuntimeError):
            cls().search(np.eye(1, 3, dtype="float32"), k=1)

    def test_one_dim_add_raises_valueerror(self, name, cls) -> None:
        # A 1-D add used to surface as an opaque AxisError deep in the backend.
        with pytest.raises(ValueError, match="2-D"):
            cls().add(np.ones(3, dtype="float32"))

    def test_non_finite_embedding_raises_valueerror(self, name, cls) -> None:
        # NaN/inf embeddings silently produced garbage scores before validation.
        store = cls()
        bad = np.ones((3, 3), dtype="float32")
        bad[1, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            store.add(bad)

    def test_query_dim_mismatch_raises_valueerror(self, name, cls) -> None:
        store = cls()
        store.add(_orthonormal_corpus())  # dim 3
        with pytest.raises(ValueError, match="dim"):
            store.search(np.ones((1, 5), dtype="float32"), k=1)
