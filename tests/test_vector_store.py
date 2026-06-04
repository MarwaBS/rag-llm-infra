"""VectorStoreProtocol conformance + known-answer search across backends.

NumPy is always exercised; FAISS and Qdrant tests skip cleanly when those
optional backends are not installed.
"""
import numpy as np
import pytest

from vector_store import (
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
