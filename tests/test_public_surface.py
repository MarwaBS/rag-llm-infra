"""What `__all__` promises, and what the protocol requires of a backend.

Every name in `__all__` of a published package is a compatibility promise, so the
set is pinned rather than left to grow. And a protocol whose implementations hold
network clients needs a way to release them, or a caller holding the protocol
type cannot.
"""

from __future__ import annotations

from typing import Any

import pytest

import rag_llm_infra as package
from rag_llm_infra import FallbackLLM, LLMProtocol, MockBackend
from rag_llm_infra.llm_protocol import AnthropicBackend, Message, OpenAIBackend

EXPORTED = {
    "LLMProtocol", "OpenAIBackend", "AnthropicBackend", "MockBackend", "get_llm",
    "VectorStoreProtocol", "FAISSVectorStore", "NumpyVectorStore",
    "QdrantVectorStore", "get_vector_store", "FAISS_AVAILABLE", "QDRANT_AVAILABLE",
    "EmbeddingEngine",
    "configure_tracing", "get_tracer", "current_trace_context",
    "configure_logging", "llm_call",
    "groundedness", "FallbackLLM", "BudgetExhausted", "__version__",
}  # fmt: skip

LIFECYCLE = ("close", "aclose")


class Recording:
    """A backend that records which lifecycle method reached it.

    A single `closed` flag cannot tell `aclose` from `aclose` delegating to
    `close`, which is the shape the async bug took.
    """

    backend_name = "rec"
    backend_version = "1"

    def __init__(self) -> None:
        self.closed_via: str | None = None

    def invoke(self, messages: list[Message], **kwargs: Any) -> str:
        return "x"

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> str:
        return "x"

    def close(self) -> None:
        self.closed_via = "sync"

    async def aclose(self) -> None:
        self.closed_via = "async"


def test_the_public_surface_is_exactly_this() -> None:
    assert set(package.__all__) == EXPORTED


def test_every_exported_name_resolves() -> None:
    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert not missing, missing


@pytest.mark.parametrize("name", ["CONFIG", "RWLock"])
def test_the_withdrawn_names_are_not_promised(name: str) -> None:
    """Still importable from their own module; the promise is what was dropped."""
    assert name not in package.__all__
    assert not hasattr(package, name)
    import rag_llm_infra.evidence_index as evidence_index

    assert hasattr(evidence_index, name)


@pytest.mark.parametrize("method", LIFECYCLE)
def test_the_protocol_requires_a_way_to_release_the_transport(method: str) -> None:
    assert hasattr(LLMProtocol, method)


@pytest.mark.parametrize("backend", [MockBackend(), AnthropicBackend(), OpenAIBackend])
@pytest.mark.parametrize("method", LIFECYCLE)
def test_every_shipped_backend_implements_it(backend: Any, method: str) -> None:
    assert hasattr(backend, method)


def test_a_backend_missing_the_lifecycle_does_not_satisfy_the_protocol() -> None:
    """`runtime_checkable` checks method presence, which is what makes this
    assertion worth anything: without `close`, the object is not conforming."""

    class NoLifecycle:
        backend_name = "none"
        backend_version = "1"

        def invoke(self, messages: list[Message], **kwargs: Any) -> str:
            return ""

        async def ainvoke(self, messages: list[Message], **kwargs: Any) -> str:
            return ""

    assert not isinstance(NoLifecycle(), LLMProtocol)
    assert isinstance(MockBackend(), LLMProtocol)


def test_closing_the_chain_closes_every_backend_including_tripped_ones() -> None:
    backends = [Recording(), Recording()]
    chain = FallbackLLM(backends)
    chain._active = 1  # the first is tripped and must still be closed
    chain.close()
    assert [b.closed_via for b in backends] == ["sync", "sync"]


async def test_closing_the_chain_asynchronously_closes_every_backend() -> None:
    """`AsyncOpenAI.close` is a coroutine, so a chain that reached for the sync
    method here would close nothing and raise nothing."""
    backends = [Recording(), Recording()]
    chain = FallbackLLM(backends)
    chain._active = 1
    await chain.aclose()
    assert [b.closed_via for b in backends] == ["async", "async"]
