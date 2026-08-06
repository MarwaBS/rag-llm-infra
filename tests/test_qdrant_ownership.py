"""The Qdrant store owns its collection, and `size` counts what is really there.

`add()` deletes and recreates the collection. A default name makes two stores on
one endpoint share it silently. The second `add()` then destroys the first
store's vectors. The first still reports the count it wrote, and `search` sizes
its rows from that number and pads the answer with `-1` sentinels.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from rag_llm_infra import get_vector_store
from rag_llm_infra import vector_store as vs

pytestmark = pytest.mark.skipif(
    not vs.QDRANT_AVAILABLE, reason="qdrant-client not installed"
)


@pytest.fixture
def one_endpoint() -> Any:
    """Every store built inside this fixture talks to one Qdrant instance."""
    shared = vs.QdrantClient(":memory:")
    with patch.object(vs, "QdrantClient", lambda *a, **k: shared):
        yield shared


def test_the_collection_name_has_no_default() -> None:
    with pytest.raises(TypeError):
        vs.QdrantVectorStore()  # type: ignore[call-arg]


def test_the_factory_refuses_qdrant_without_a_collection() -> None:
    with pytest.raises(ValueError, match="collection="):
        get_vector_store("qdrant")


def test_the_factory_rejects_a_collection_for_the_other_backends() -> None:
    with pytest.raises(ValueError, match="applies to qdrant"):
        get_vector_store("numpy", collection="evidence")


def test_two_stores_that_named_different_collections_keep_their_data(
    one_endpoint: Any,
) -> None:
    first = get_vector_store("qdrant", collection="first")
    second = get_vector_store("qdrant", collection="second")
    first.add(np.eye(3, dtype="float32"))
    second.add(np.eye(1, 3, dtype="float32"))
    assert first.size == 3
    assert second.size == 1


def test_size_counts_the_collection_rather_than_remembering(
    one_endpoint: Any,
) -> None:
    store = vs.QdrantVectorStore(collection="owned")
    store.add(np.eye(3, dtype="float32"))
    one_endpoint.delete_collection(collection_name="owned")
    one_endpoint.create_collection(
        collection_name="owned",
        vectors_config=vs.qdrant_models.VectorParams(
            size=3, distance=vs.qdrant_models.Distance.COSINE
        ),
    )
    assert store.size == 0


def test_search_never_pads_with_sentinels_when_the_store_shrinks(
    one_endpoint: Any,
) -> None:
    """Row width is `min(k, size)`, so it has to come from the live count."""
    store = vs.QdrantVectorStore(collection="owned")
    store.add(np.eye(3, dtype="float32"))
    store.add(np.eye(1, 3, dtype="float32"))
    _, idx = store.search(np.eye(1, 3, dtype="float32"), k=3)
    assert idx.shape == (1, 1)
    assert -1 not in idx[0].tolist()
