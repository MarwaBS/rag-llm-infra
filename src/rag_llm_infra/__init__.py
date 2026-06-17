"""rag-llm-infra — vendor-neutral RAG + LLM serving infrastructure.

Public API: the LLM-provider and vector-store protocols + factories, the cached
embedding index, and the observability helpers.
"""

from __future__ import annotations

from .evidence_index import CONFIG, EmbeddingEngine, RWLock
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

__version__ = "0.1.2"

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
    "RWLock",
    "CONFIG",
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
