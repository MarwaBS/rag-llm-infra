"""Minimal FastAPI service exposing the RAG infra: index documents, then query.

    pip install "rag-llm-infra[serve]"
    uvicorn rag_llm_infra.serve:app

Runs on the NumPy vector store + a deterministic demo embedder + the Mock LLM, so
it needs no API key. For production, swap the demo embedder for `EmbeddingEngine`
and `get_llm("mock")` for `get_llm("openai")`.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from . import get_llm, get_vector_store
from ._demo import embed

app = FastAPI(title="rag-llm-infra", version="0.1.0")
_state: Dict[str, Any] = {"docs": [], "store": None}


class IndexRequest(BaseModel):
    documents: List[str]


class QueryRequest(BaseModel):
    query: str
    k: int = 3


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/index")
def index(req: IndexRequest) -> Dict[str, int]:
    store = get_vector_store("numpy")
    store.add(embed(req.documents))
    _state["docs"], _state["store"] = req.documents, store
    return {"indexed": len(req.documents)}


@app.post("/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    store, docs = _state["store"], _state["docs"]
    if store is None:
        return {"error": "index documents first", "retrieved": [], "answer": ""}
    _, idx = store.search(embed([req.query]), k=min(req.k, len(docs)))
    retrieved = [docs[int(i)] for i in idx[0] if i >= 0]
    context = "\n".join(f"- {d}" for d in retrieved)
    llm = get_llm("mock", response=lambda _m: f"(answer grounded in {len(retrieved)} retrieved docs)")
    answer = llm.invoke([
        {"role": "system", "content": "Answer using ONLY the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
    ])
    return {"retrieved": retrieved, "answer": answer}
