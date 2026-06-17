"""Smoke + error-surface tests for the FastAPI serving layer (no network)."""

import pytest
from fastapi.testclient import TestClient

import rag_llm_infra.serve as serve
from rag_llm_infra.serve import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_index():
    serve._index = None
    yield
    serve._index = None


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_index_then_query_grounds_answer() -> None:
    docs = [
        "FAISS performs in-process vector similarity search.",
        "Qdrant is a vector database with REST and gRPC APIs.",
    ]
    r = client.post("/index", json={"documents": docs})
    assert r.status_code == 201
    assert r.json() == {"indexed": 2}
    body = client.post(
        "/query", json={"query": "vector similarity search", "k": 1}
    ).json()
    assert body["retrieved"] == ["FAISS performs in-process vector similarity search."]
    assert "grounded" in body["answer"]


def test_query_before_index_returns_409() -> None:
    r = client.post("/query", json={"query": "anything"})
    assert r.status_code == 409
    assert r.json()["error"] == "index documents first"


def test_index_rejects_empty_documents_422() -> None:
    assert client.post("/index", json={"documents": []}).status_code == 422


def test_query_rejects_nonpositive_k_422() -> None:
    client.post("/index", json={"documents": ["a doc about vectors"]})
    for bad_k in (0, -3):
        assert (
            client.post("/query", json={"query": "vectors", "k": bad_k}).status_code
            == 422
        )


def test_reindex_with_smaller_corpus_never_500s() -> None:
    client.post("/index", json={"documents": [f"doc {i} vectors" for i in range(15)]})
    client.post("/index", json={"documents": ["only one vectors doc"]})
    r = client.post("/query", json={"query": "vectors", "k": 5})
    assert r.status_code == 200
    assert r.json()["retrieved"] == ["only one vectors doc"]
