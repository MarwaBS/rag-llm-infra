"""`reset()` clears the store even when the backend refuses to co-operate.

Both backends catch a failure from the underlying index and log it. Untested,
those branches were two of the six `# pragma: no cover` exclusions worth 2.89
points of the coverage figure. Testing them is cheaper than explaining them.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from rag_llm_infra.vector_store import (
    FAISS_AVAILABLE,
    QDRANT_AVAILABLE,
    FAISSVectorStore,
    QdrantVectorStore,
)


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
def test_faiss_reset_survives_an_index_that_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = FAISSVectorStore()
    store.add(np.eye(3, dtype="float32"))

    def refuse() -> None:
        raise RuntimeError("index refused")

    monkeypatch.setattr(store._index, "reset", refuse)
    with caplog.at_level(logging.WARNING):
        store.reset()

    assert store.size == 0, "reset left the store holding vectors"
    assert any("reset failed" in r.message for r in caplog.records), caplog.text


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
def test_qdrant_reset_survives_a_client_that_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = QdrantVectorStore(collection="reset-probe")
    store.add(np.eye(3, dtype="float32"))

    def refuse(**kwargs: object) -> None:
        raise RuntimeError("collection refused")

    monkeypatch.setattr(store._client, "delete_collection", refuse)
    with caplog.at_level(logging.WARNING):
        store.reset()

    assert store.size == 0
    assert any("reset failed" in r.message for r in caplog.records), caplog.text


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
def test_the_control_shows_a_clean_reset_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = FAISSVectorStore()
    store.add(np.eye(3, dtype="float32"))
    with caplog.at_level(logging.WARNING):
        store.reset()
    assert store.size == 0
    assert not [r for r in caplog.records if "reset failed" in r.message]
