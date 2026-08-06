"""The `-1` sentinel is part of the Qdrant contract, and every consumer filters it.

A batch answers as one rectangular array, so a short row is padded. A caller who
reads the protocol docstring and indexes `docs[-1]` silently gets the last
document instead of a miss.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rag_llm_infra import vector_store as vs

pytestmark = pytest.mark.skipif(
    not vs.QDRANT_AVAILABLE, reason="qdrant-client not installed"
)


def test_a_short_row_is_padded_with_minus_one(monkeypatch: pytest.MonkeyPatch) -> None:
    store = vs.QdrantVectorStore(collection="padding")
    store.add(np.eye(3, dtype="float32"))

    def one_hit(collection_name, requests):  # noqa: ANN001, ANN202
        return [SimpleNamespace(points=[SimpleNamespace(score=1.0, id=0)])] * len(
            requests
        )

    monkeypatch.setattr(store._client, "query_batch_points", one_hit)
    scores, idx = store.search(np.eye(1, 3, dtype="float32"), k=3)
    assert idx[0].tolist() == [0, -1, -1]
    assert scores[0].tolist() == [1.0, -1.0, -1.0]


def test_the_control_shows_a_full_row_is_not_padded() -> None:
    store = vs.QdrantVectorStore(collection="padding")
    store.add(np.eye(3, dtype="float32"))
    _, idx = store.search(np.eye(1, 3, dtype="float32"), k=3)
    assert -1 not in idx[0].tolist()


REPO = Path(__file__).resolve().parent.parent
LOOKUP = re.compile(r"^.*\bfor \w+ in (?:idx|indices)\[0\].*$", re.M)


def _sources() -> list[Path]:
    """Every tracked Python file, not a list somebody maintains by hand.

    A hand-written list is a blacklist over an open set: the next module that
    turns indices into documents is not on it.
    """
    listed = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPO / rel for rel in listed if not rel.startswith("tests/")]


def test_the_sweep_finds_the_lookups_it_is_meant_to_check() -> None:
    found = {p.name for p in _sources() if LOOKUP.search(p.read_text(encoding="utf-8"))}
    assert {
        "serve.py",
        "example.py",
        "generation_eval.py",
        "retrieval_eval.py",
    } <= found


def test_every_index_lookup_outside_the_tests_filters_the_sentinel() -> None:
    """An unfiltered `-1` indexes the last document and returns it as a match."""
    unfiltered = [
        f"{path.name}: {line.strip()}"
        for path in _sources()
        for line in LOOKUP.findall(path.read_text(encoding="utf-8"))
        if ">= 0" not in line
    ]
    assert not unfiltered, unfiltered
