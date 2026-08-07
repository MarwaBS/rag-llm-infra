"""rag-llm-infra: vendor-neutral RAG + LLM serving infrastructure.

Public API: the LLM-provider and vector-store protocols + factories, the cached
embedding index, and the observability helpers.

`CONFIG` and `RWLock` are not exported. Everything named here is a compatibility
promise, and neither could be kept: `CONFIG` is read once, when an
`EmbeddingEngine` is constructed, so a caller who mutates it afterwards changes
nothing and has no way to tell. `RWLock` is an implementation detail.
Both remain reachable at `rag_llm_infra.evidence_index` for anyone who accepts
that they can change.
"""

from __future__ import annotations

from .evidence_index import EmbeddingEngine
from .faithfulness import groundedness
from .fallback import BudgetExhausted, FallbackLLM
from .llm_protocol import (
    AnthropicBackend,
    LLMProtocol,
    MockBackend,
    OpenAIBackend,
    get_llm,
)
from .log_config import configure_logging, llm_call
from .tracing import configure_tracing, current_trace_context, get_tracer
from .vector_store import (
    FAISS_AVAILABLE,
    QDRANT_AVAILABLE,
    FAISSVectorStore,
    NumpyVectorStore,
    QdrantVectorStore,
    VectorStoreProtocol,
    get_vector_store,
)

__version__ = "0.2.0"

__all__ = [
    # LLM
    "LLMProtocol",
    "OpenAIBackend",
    "AnthropicBackend",
    "MockBackend",
    "get_llm",
    # Vector store
    "VectorStoreProtocol",
    "FAISSVectorStore",
    "NumpyVectorStore",
    "QdrantVectorStore",
    "get_vector_store",
    "FAISS_AVAILABLE",
    "QDRANT_AVAILABLE",
    # Embedding index
    "EmbeddingEngine",
    # Observability
    "configure_tracing",
    "get_tracer",
    "current_trace_context",
    "configure_logging",
    "llm_call",
    # Faithfulness + fallback
    "groundedness",
    "FallbackLLM",
    "BudgetExhausted",
    "__version__",
]
