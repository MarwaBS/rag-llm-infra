"""Minimal FastAPI service exposing the RAG infra: index documents, then query.

    pip install "rag-llm-infra[serve]"
    uvicorn rag_llm_infra.serve:app

Runs on the NumPy vector store + a deterministic demo embedder + the Mock LLM, so
it needs no API key. The corpus is held in process (single replica). For
production, swap the demo embedder for `EmbeddingEngine` and `get_llm("mock")` for
`get_llm("openai")`.

`/index` and `/query` are unauthenticated: anyone who reaches the port can
replace the corpus and read it back. Put it behind your own auth before exposing
it.

This module configures neither logging nor tracing. `configure_logging()` and
`configure_tracing()` are yours to call. The command above hands the import to
uvicorn, leaving no point to call them first, so call them from your own module
and point uvicorn at that instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__, get_llm, get_vector_store
from ._demo import embed

# Sourced from the package so the served version cannot drift from the wheel's.
app = FastAPI(title="rag-llm-infra", version=__version__)


@dataclass(frozen=True)
class _Index:
    """Immutable (docs, store) snapshot, swapped atomically. One reference means
    /query cannot read a new store paired with stale docs and IndexError."""

    docs: tuple[str, ...]
    store: Any


_index: _Index | None = None


class IndexRequest(BaseModel):
    documents: list[str] = Field(min_length=1)  # empty corpus is meaningless


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=3, ge=1)  # a non-positive k otherwise reaches the store


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index", status_code=201)
def index(req: IndexRequest) -> dict[str, int]:
    store = get_vector_store("numpy")
    store.add(embed(list(req.documents)))
    global _index
    _index = _Index(docs=tuple(req.documents), store=store)
    return {"indexed": len(req.documents)}


@app.post("/query")
def query(req: QueryRequest) -> Any:
    snapshot = _index  # single atomic read
    if snapshot is None:
        # Query before any corpus exists is a client error, not a 200 body with
        # an "error" key.
        return JSONResponse(
            {"error": "index documents first", "retrieved": [], "answer": ""},
            status_code=409,
        )
    docs, store = snapshot.docs, snapshot.store
    _, idx = store.search(embed([req.query]), k=min(req.k, len(docs)))
    retrieved = [docs[int(i)] for i in idx[0] if i >= 0]
    context = "\n".join(f"- {d}" for d in retrieved)
    llm = get_llm(
        "mock",
        response=lambda _m: f"(answer grounded in {len(retrieved)} retrieved docs)",
    )
    answer = llm.invoke(
        [
            {"role": "system", "content": "Answer using ONLY the provided context."},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {req.query}",
            },
        ]
    )
    return {"retrieved": retrieved, "answer": answer}
