"""The `-1` sentinel is part of the Qdrant contract, and every consumer filters it.

A batch answers as one rectangular array, so a short row is padded. A caller who
reads the protocol docstring and indexes `docs[-1]` silently gets the last
document instead of a miss.
"""

from __future__ import annotations

import ast
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
INDEX_NAMES = {"idx", "indices"}


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


def _over_a_result_row(comprehension: ast.comprehension) -> bool:
    """Whether this iterates the first row of a `search` result."""
    iterated = comprehension.iter
    return (
        isinstance(iterated, ast.Subscript)
        and isinstance(iterated.value, ast.Name)
        and iterated.value.id in INDEX_NAMES
    )


def _int_literal(node: ast.expr) -> int | None:
    """`-1` parses as a unary minus over `1`, not as a negative constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _int_literal(node.operand)
        return None if inner is None else -inner
    return None


def _excludes_negatives(comprehension: ast.comprehension) -> bool:
    """Whether a condition rules out a negative value of the loop variable.

    Read as a comparison against the loop target, not as text. `">= 0" in line`
    credits `if len(docs) >= 0`, which filters nothing, and rejects `if i > -1`,
    which filters correctly.
    """
    if not isinstance(comprehension.target, ast.Name):
        return False
    target = comprehension.target.id
    for condition in comprehension.ifs:
        if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
            continue
        left, operator, right = (
            condition.left,
            condition.ops[0],
            condition.comparators[0],
        )
        named_left = isinstance(left, ast.Name) and left.id == target
        named_right = isinstance(right, ast.Name) and right.id == target
        floor = _int_literal(right if named_left else left)
        if floor is None:
            continue
        if named_left and (
            (isinstance(operator, ast.GtE) and floor == 0)
            or (isinstance(operator, ast.Gt) and floor == -1)
        ):
            return True
        if named_right and (
            (isinstance(operator, ast.LtE) and floor == 0)
            or (isinstance(operator, ast.Lt) and floor == -1)
        ):
            return True
    return False


def _row_comprehensions() -> list[tuple[str, ast.comprehension]]:
    found = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for comprehension in getattr(node, "generators", []):
                if _over_a_result_row(comprehension):
                    found.append((path.name, comprehension))
    return found


def test_the_sweep_finds_the_lookups_it_is_meant_to_check() -> None:
    found = {name for name, _ in _row_comprehensions()}
    assert {
        "serve.py",
        "example.py",
        "generation_eval.py",
        "retrieval_eval.py",
    } <= found


def test_every_index_lookup_outside_the_tests_filters_the_sentinel() -> None:
    """An unfiltered `-1` indexes the last document and returns it as a match."""
    unfiltered = [
        name
        for name, comprehension in _row_comprehensions()
        if not _excludes_negatives(comprehension)
    ]
    assert not unfiltered, unfiltered
