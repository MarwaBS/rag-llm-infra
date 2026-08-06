"""Minimal FastAPI service exposing the RAG infra: index documents, then query.

    pip install "rag-llm-infra[serve]"
    uvicorn rag_llm_infra.serve:app

Runs on the NumPy vector store, a deterministic demo embedder and the Mock LLM,
so it needs no *provider* credential. The corpus is held in process (single
replica). For production, swap the demo embedder for `EmbeddingEngine` and
`get_llm("mock")` for `get_llm("openai")`.

`/index` and `/query` require `X-API-Key` to match `RAG_API_KEY`. With that
variable unset they answer 503 rather than running open. There is no
configuration in which the corpus is replaceable by anyone who reaches the port.
`/health` stays open for container probes and reveals nothing but liveness.

One size bound, because the corpus is held in this process. A request body over
`RAG_MAX_BODY_BYTES` (1 MiB by default) is refused with 413, and a POST without
a `Content-Length` with 411. Document count and length need no separate limits —
they cannot exceed what the body bound admits. `k` is already capped at the
corpus size.

This module configures neither logging nor tracing. The command above hands the
import to uvicorn, so call `configure_logging()` / `configure_tracing()` from
your own module and point uvicorn at that.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__, get_llm, get_vector_store
from ._demo import embed

DEFAULT_MAX_BODY_BYTES = 1024 * 1024

# Sourced from the package so the served version cannot drift from the wheel's.
app = FastAPI(title="rag-llm-infra", version=__version__)


def _max_body_bytes() -> int:
    """Read per request, so a deployment can change it without a restart."""
    return int(os.getenv("RAG_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES))


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """401 without the key, 503 when the server has none configured.

    Answering 503 rather than serving is what stops an operator who forgot the
    variable from publishing an open corpus.
    """
    expected = os.getenv("RAG_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="RAG_API_KEY is not configured")
    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.middleware("http")
async def bound_request_body(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Refuse an oversized body before it is read into memory."""
    if request.method in ("POST", "PUT", "PATCH"):
        declared = request.headers.get("content-length")
        if declared is None:
            return JSONResponse({"detail": "Content-Length required"}, status_code=411)
        if int(declared) > _max_body_bytes():
            return JSONResponse({"detail": "request body too large"}, status_code=413)
    return await call_next(request)


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


@app.post("/index", status_code=201, dependencies=[Depends(require_api_key)])
def index(req: IndexRequest) -> dict[str, int]:
    store = get_vector_store("numpy")
    store.add(embed(list(req.documents)))
    global _index
    _index = _Index(docs=tuple(req.documents), store=store)
    return {"indexed": len(req.documents)}


@app.post("/query", dependencies=[Depends(require_api_key)])
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
