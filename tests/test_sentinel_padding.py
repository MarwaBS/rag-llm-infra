"""The `-1` sentinel is part of the Qdrant contract, and every consumer filters it.

A batch answers as one rectangular array, so a short row is padded. A caller who
reads the protocol docstring and indexes `docs[-1]` silently gets the last
document instead of a miss.
"""

from __future__ import annotations

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


@pytest.mark.parametrize(
    "module",
    ["rag_llm_infra.serve", "example", "eval.generation_eval", "eval.retrieval_eval"],
)
def test_every_module_that_turns_indices_into_documents_filters(module: str) -> None:
    """An unfiltered `-1` indexes the last document and returns it as a match."""
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module))
    lookups = [
        ln
        for ln in source.splitlines()
        if "for i in idx" in ln or "in indices[0]" in ln
    ]
    assert lookups, f"{module}: no index lookup found"
    unfiltered = [ln.strip() for ln in lookups if ">= 0" not in ln]
    assert not unfiltered, f"{module} does not filter the sentinel: {unfiltered}"
