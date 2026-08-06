"""The demo embedder's defining properties, pinned.

Both eval gates, `example.py` and the serving demo embed through this function,
so every retrieval number they report is a statement about it. Case folding,
term frequency and cross-process reproducibility are the three the scores do not
move enough to catch on their own.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from rag_llm_infra._demo import EMBED_DIM, embed

_TEXT = "FAISS Qdrant OpenTelemetry"


def test_the_shape_and_dtype_are_the_documented_ones() -> None:
    vecs = embed(["one", "two"])
    assert vecs.shape == (2, EMBED_DIM)
    assert vecs.dtype == np.dtype("float32")


def test_case_does_not_change_the_vector() -> None:
    assert np.array_equal(embed([_TEXT])[0], embed([_TEXT.lower()])[0])


def test_the_control_shows_different_text_does_move_the_vector() -> None:
    assert not np.array_equal(embed([_TEXT])[0], embed(["cartography"])[0])


def test_digits_are_content() -> None:
    """The token pattern is `[a-z0-9]+`. Without the digits a purely numeric
    token disappears and "v2" and "v3" become the same token."""
    assert embed(["2024"])[0].sum() == 1.0
    assert not np.array_equal(embed(["v2"])[0], embed(["v3"])[0])


def test_a_repeated_token_weighs_more_than_a_single_one() -> None:
    once = embed(["faiss"])[0]
    twice = embed(["faiss faiss"])[0]
    assert twice.sum() == 2 * once.sum()
    assert not np.array_equal(once, twice)


def test_the_vector_is_the_same_in_a_fresh_interpreter() -> None:
    """The built-in `hash()` is salted per process; `hashlib` is not. Both eval
    gates read floors derived under a different interpreter than the one that
    checks them."""
    script = (
        f"from rag_llm_infra._demo import embed; print(embed([{_TEXT!r}])[0].tolist())"
    )
    seen = set()
    for seed in ("0", "1", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        seen.add(proc.stdout.strip())
    assert len(seen) == 1, "the embedding changed with the hash seed"
    assert seen.pop() == str(embed([_TEXT])[0].tolist())
