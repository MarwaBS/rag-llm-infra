"""What the search contract promises about equal scores.

Duplicate and near-duplicate documents are ordinary in a RAG corpus, so ties are
reachable rather than theoretical. Every backend must repeat itself; only NumPy
promises which tied document comes first.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest

from rag_llm_infra.vector_store import (
    FAISS_AVAILABLE,
    QDRANT_AVAILABLE,
    FAISSVectorStore,
    NumpyVectorStore,
    QdrantVectorStore,
    VectorStoreProtocol,
)

QUERY = np.array([[1.0, 0.0, 0.0]], dtype="float32")


def _tied_corpus(n: int = 8) -> np.ndarray:
    return np.tile(np.array([1.0, 0.0, 0.0], dtype="float32"), (n, 1))


def _builders() -> list[tuple[str, Callable[[], VectorStoreProtocol]]]:
    items: list[tuple[str, Callable[[], VectorStoreProtocol]]] = [
        ("numpy", NumpyVectorStore)
    ]
    if FAISS_AVAILABLE:
        items.append(("faiss", FAISSVectorStore))
    if QDRANT_AVAILABLE:
        items.append(("qdrant", lambda: QdrantVectorStore(collection="ties")))
    return items


@pytest.mark.parametrize("n,k", [(8, 4), (100, 3), (1000, 256), (1000, 999), (37, 37)])
def test_the_ordering_holds_at_every_size(n: int, k: int) -> None:
    """The promise is the order of what comes back, not which documents come
    back: `argpartition` picks the candidates, and the contract calls that
    selection unspecified when more than k share the boundary score.

    The wide cases carry the weight. NumPy's introsort drops to insertion sort
    on short rows, so a small tie is ordered correctly by accident and cannot
    tell a stable sort from an unstable one.
    """
    store = NumpyVectorStore()
    store.add(_tied_corpus(n))
    _, idx = store.search(QUERY, k=k)
    returned = [int(i) for i in idx[0]]
    assert returned == sorted(returned)
    assert len(returned) == min(k, n)


@pytest.mark.parametrize("name,build", _builders())
def test_every_backend_repeats_itself_within_a_process(name: str, build) -> None:
    first = build()
    first.add(_tied_corpus())
    second = build()
    second.add(_tied_corpus())
    _, a = first.search(QUERY, k=4)
    _, b = second.search(QUERY, k=4)
    assert a[0].tolist() == b[0].tolist(), name


def test_numpy_repeats_itself_across_processes() -> None:
    """A fresh interpreter, because a hash seed or a partition detail that
    varied per process would not show up in a single one."""
    script = (
        "import numpy as np;"
        "from rag_llm_infra.vector_store import NumpyVectorStore as S;"
        "s=S();s.add(np.tile(np.array([1.,0.,0.],dtype='float32'),(8,1)));"
        "print([int(i) for i in s.search(np.array([[1.,0.,0.]],dtype='float32'),k=4)[1][0]])"
    )
    seen = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(seen) == 1, seen


def test_the_control_shows_ordering_still_follows_the_score() -> None:
    """Tie-breaking must not outrank the score itself."""
    store = NumpyVectorStore()
    store.add(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype="float32"))
    scores, idx = store.search(QUERY, k=2)
    assert [int(i) for i in idx[0]] == [1, 0]
    assert scores[0][0] > scores[0][1]
