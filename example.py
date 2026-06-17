"""
End-to-end retrieval-augmented generation, wired from this package's parts.

    embed documents → index in a VectorStore → retrieve top-k for a query
                     → build a grounded prompt → answer with an LLMProtocol backend

Runs on the NumPy vector store and the deterministic mock LLM, so it needs no
API key, no network, and no native libraries:

    pip install rag-llm-infra
    python example.py

In production, swap the demo embedder for `rag_llm_infra.EmbeddingEngine` (real
sentence embeddings) and `get_llm("mock")` for `get_llm("openai")`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from rag_llm_infra import get_llm, get_vector_store
from rag_llm_infra._demo import embed

if TYPE_CHECKING:
    from rag_llm_infra import LLMProtocol


def retrieve(docs: List[str], query: str, k: int = 2) -> List[str]:
    """Index `docs` and return the top-`k` most similar to `query`."""
    store = get_vector_store("numpy")  # always available; no native deps
    store.add(embed(docs))
    _, indices = store.search(embed([query]), k=min(k, len(docs)))
    return [docs[int(i)] for i in indices[0] if i >= 0]


def rag_answer(docs: List[str], query: str, llm: "Optional[LLMProtocol]" = None) -> str:
    """Retrieve grounding context, then answer the query with an LLM backend."""
    context = retrieve(docs, query)
    grounded = "\n".join(f"- {d}" for d in context)
    messages = [
        {"role": "system", "content": "Answer using ONLY the provided context."},
        {"role": "user", "content": f"Context:\n{grounded}\n\nQuestion: {query}"},
    ]
    llm = llm or get_llm(
        "mock",
        response=lambda _m: f"(mock answer grounded in {len(context)} retrieved docs)",
    )
    return llm.invoke(messages)


DOCS = [
    "FAISS is an in-process library for fast vector similarity search.",
    "Qdrant is a vector database offering REST and gRPC APIs.",
    "OpenTelemetry provides vendor-neutral distributed tracing.",
    "Retrieval-augmented generation grounds an LLM in retrieved documents.",
]


def main() -> None:
    query = "Which approach grounds an LLM in retrieved documents?"
    print("Query:    ", query)
    print("Retrieved:", retrieve(DOCS, query))
    print("Answer:   ", rag_answer(DOCS, query))


if __name__ == "__main__":
    main()
