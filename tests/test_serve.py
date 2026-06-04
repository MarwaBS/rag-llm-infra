"""Smoke tests for the FastAPI serving layer (index -> query, no network)."""
from fastapi.testclient import TestClient

from rag_llm_infra.serve import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_index_then_query_grounds_answer() -> None:
    docs = [
        "FAISS performs in-process vector similarity search.",
        "Qdrant is a vector database with REST and gRPC APIs.",
    ]
    assert client.post("/index", json={"documents": docs}).json() == {"indexed": 2}
    body = client.post("/query", json={"query": "vector similarity search", "k": 1}).json()
    assert body["retrieved"] == ["FAISS performs in-process vector similarity search."]
    assert "grounded" in body["answer"]
