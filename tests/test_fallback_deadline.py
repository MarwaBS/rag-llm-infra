"""A backend that hangs must not hold the chain.

`FallbackLLM` advances when a backend raises. A provider that blocks raises
nothing, which is the failure mode the module exists to survive.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from rag_llm_infra.fallback import BackendTimeout, FallbackLLM

MESSAGES = [{"role": "user", "content": "hi"}]


class _Backend:
    def __init__(self, name: str, delay: float = 0.0) -> None:
        self.backend_name = name
        self._delay = delay
        self.started = threading.Event()

    def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.started.set()
        time.sleep(self._delay)
        return f"{self.backend_name} answer"

    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.started.set()
        await asyncio.sleep(self._delay)
        return f"{self.backend_name} answer"


def test_a_blocking_backend_does_not_hold_the_chain() -> None:
    slow = _Backend("slow", delay=3.0)
    chain = FallbackLLM([slow, _Backend("fast")], timeout_s=0.2)
    start = time.perf_counter()
    answer = chain.invoke(MESSAGES)
    elapsed = time.perf_counter() - start
    assert answer == "fast answer"
    assert slow.started.is_set(), "the slow backend was never tried"
    assert elapsed < 1.5, f"waited {elapsed:.2f}s on a 0.2s deadline"


def test_without_a_deadline_the_chain_waits() -> None:
    """The control: the deadline is what advances the chain, not the ordering."""
    chain = FallbackLLM([_Backend("slow", delay=0.3), _Backend("fast")])
    assert chain.invoke(MESSAGES) == "slow answer"


def test_the_deadline_does_not_delay_a_backend_that_answers() -> None:
    chain = FallbackLLM([_Backend("fast")], timeout_s=5.0)
    start = time.perf_counter()
    assert chain.invoke(MESSAGES) == "fast answer"
    assert time.perf_counter() - start < 1.0


def test_every_backend_timing_out_raises_rather_than_hanging() -> None:
    chain = FallbackLLM(
        [_Backend("a", delay=3.0), _Backend("b", delay=3.0)], timeout_s=0.2
    )
    with pytest.raises(RuntimeError, match="all 2 backends failed"):
        chain.invoke(MESSAGES)


def test_a_failure_raised_inside_the_worker_reaches_the_caller() -> None:
    """The deadline runs the call on another thread; its exception must still
    drive the chain rather than being swallowed."""

    class Broken:
        backend_name = "broken"

        def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            raise ConnectionError("provider down")

        async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            raise ConnectionError("provider down")

    chain = FallbackLLM([Broken(), _Backend("fast")], timeout_s=5.0)
    assert chain.invoke(MESSAGES) == "fast answer"


def test_a_non_retryable_error_still_propagates_through_the_deadline() -> None:
    class Bug:
        backend_name = "bug"

        def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            raise TypeError("a bug, not a provider failure")

        async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            raise TypeError("a bug, not a provider failure")

    chain = FallbackLLM([Bug(), _Backend("fast")], timeout_s=5.0)
    with pytest.raises(TypeError):
        chain.invoke(MESSAGES)


async def test_the_async_path_advances_on_its_deadline() -> None:
    slow = _Backend("slow", delay=3.0)
    chain = FallbackLLM([slow, _Backend("fast")], timeout_s=0.2)
    start = time.perf_counter()
    assert await chain.ainvoke(MESSAGES) == "fast answer"
    assert time.perf_counter() - start < 1.5


def test_a_non_positive_deadline_is_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        FallbackLLM([_Backend("a")], timeout_s=0)


def test_the_timeout_type_is_retryable() -> None:
    assert issubclass(BackendTimeout, Exception)
