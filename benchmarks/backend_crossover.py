"""Where FAISS starts beating NumPy, on the machine that runs this.

    python -m benchmarks.backend_crossover

`get_vector_store("auto")` prefers FAISS whenever it imports. That preference had
no measurement behind it. This times `search` on both backends across corpus
sizes so the choice is a number rather than an assumption.

Bounded: one machine, one build of faiss-cpu, brute-force inner product over
random vectors. It says where the crossover is here, not on your production
hardware, and it says nothing about recall — both backends return the same
answers, which `tests/test_vector_store.py` pins.
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np

from rag_llm_infra.vector_store import (
    FAISS_AVAILABLE,
    FAISSVectorStore,
    NumpyVectorStore,
)

SIZES = (100, 1_000, 10_000, 100_000)
DIM = 384  # all-MiniLM-L6-v2, the model EmbeddingEngine loads by default
K = 5
QUERIES = 10
REPEATS = 5


def _time_search(store: object, corpus: np.ndarray, queries: np.ndarray) -> float:
    """Milliseconds per search call, excluding indexing."""
    store.add(corpus)  # type: ignore[attr-defined]
    store.search(queries, k=K)  # type: ignore[attr-defined]
    start = time.perf_counter()
    for _ in range(REPEATS):
        store.search(queries, k=K)  # type: ignore[attr-defined]
    return (time.perf_counter() - start) / REPEATS * 1000


def main() -> int:
    if not FAISS_AVAILABLE:
        print("faiss is not importable here; nothing to compare.")
        return 0

    print(
        f"{platform.python_implementation()} {platform.python_version()} "
        f"numpy {np.__version__} on {platform.platform()}"
    )
    print(f"dim={DIM}, k={K}, {QUERIES} queries per call, {REPEATS} repeats\n")
    print(f"{'documents':>10} {'numpy ms':>10} {'faiss ms':>10} {'faster':>10}")

    rng = np.random.default_rng(0)
    for n in SIZES:
        corpus = rng.random((n, DIM), dtype=np.float32)
        queries = rng.random((QUERIES, DIM), dtype=np.float32)
        numpy_ms = _time_search(NumpyVectorStore(), corpus, queries)
        faiss_ms = _time_search(FAISSVectorStore(), corpus.copy(), queries)
        winner = "faiss" if faiss_ms < numpy_ms else "numpy"
        print(f"{n:>10} {numpy_ms:>10.2f} {faiss_ms:>10.2f} {winner:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
