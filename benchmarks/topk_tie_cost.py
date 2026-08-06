"""What a fully deterministic top-k would cost.

    python -m benchmarks.topk_tie_cost

`NumpyVectorStore.search` partitions for the top-k and then orders that slice by
score, breaking ties on the lower document index. It does not decide *which*
tied documents are selected. A full `lexsort` would decide both, and this
measures the difference so the choice is a number rather than an opinion.

Prints the host it ran on, because a ratio without a machine is not reproducible.
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np

SIZES = (1_000, 10_000, 100_000, 1_000_000)
K = 5
REPEATS = 5


def partition_then_order(similarities: np.ndarray, k: int) -> np.ndarray:
    """What `NumpyVectorStore.search` does."""
    k_eff = min(k, similarities.shape[1])
    top = np.argpartition(-similarities, k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(similarities.shape[0])[:, None]
    scores = similarities[rows, top]
    return np.take_along_axis(top, np.lexsort((top, -scores), axis=1), axis=1)


def fully_sorted(similarities: np.ndarray, k: int) -> np.ndarray:
    """Selection and order both decided, at the cost of sorting everything."""
    k_eff = min(k, similarities.shape[1])
    indices = np.arange(similarities.shape[1])
    order = np.lexsort(
        (np.broadcast_to(indices, similarities.shape), -similarities), axis=1
    )
    return order[:, :k_eff]


def _time(fn, similarities: np.ndarray) -> float:
    fn(similarities, K)
    start = time.perf_counter()
    for _ in range(REPEATS):
        fn(similarities, K)
    return (time.perf_counter() - start) / REPEATS * 1000


def main() -> int:
    print(
        f"{platform.python_implementation()} {platform.python_version()} "
        f"numpy {np.__version__} on {platform.platform()}"
    )
    print(f"k={K}, {REPEATS} repeats, milliseconds per call\n")
    print(
        f"{'documents':>10} {'scores':>10} {'partition':>10} {'full sort':>10} {'ratio':>7}"
    )
    rng = np.random.default_rng(0)
    for n in SIZES:
        for label, similarities in (
            ("distinct", rng.random((1, n), dtype="float32")),
            ("all tied", np.zeros((1, n), dtype="float32")),
        ):
            fast = _time(partition_then_order, similarities)
            slow = _time(fully_sorted, similarities)
            print(
                f"{n:>10} {label:>10} {fast:>10.2f} {slow:>10.2f} {slow / fast:>6.1f}x"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
