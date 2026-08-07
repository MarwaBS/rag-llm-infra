"""A backend that hangs must not hold the chain.

`FallbackLLM` advances when a backend raises. A provider that blocks raises
nothing, which is the failure mode the module exists to survive.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from rag_llm_infra.fallback import BudgetExhausted, FallbackLLM
from rag_llm_infra.llm_protocol import Message

MESSAGES: list[Message] = [{"role": "user", "content": "hi"}]


class _Backend:
    backend_version = "0"

    def __init__(self, name: str, delay: float = 0.0) -> None:
        self.backend_name = name
        self._delay = delay
        self.started = threading.Event()

    def invoke(self, messages: list[Message], **kwargs: Any) -> str:
        self.started.set()
        time.sleep(self._delay)
        return f"{self.backend_name} answer"

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> str:
        self.started.set()
        await asyncio.sleep(self._delay)
        return f"{self.backend_name} answer"

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


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
        backend_version = "0"

        def invoke(self, messages: list[Message], **kwargs: Any) -> str:
            raise ConnectionError("provider down")

        async def ainvoke(self, messages: list[Message], **kwargs: Any) -> str:
            raise ConnectionError("provider down")

        def close(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    chain = FallbackLLM([Broken(), _Backend("fast")], timeout_s=5.0)
    assert chain.invoke(MESSAGES) == "fast answer"


def test_a_non_retryable_error_still_propagates_through_the_deadline() -> None:
    class Bug:
        backend_name = "bug"
        backend_version = "0"

        def invoke(self, messages: list[Message], **kwargs: Any) -> str:
            raise TypeError("a bug, not a provider failure")

        async def ainvoke(self, messages: list[Message], **kwargs: Any) -> str:
            raise TypeError("a bug, not a provider failure")

        def close(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

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


def test_a_narrowed_retry_on_cannot_strand_the_chain_on_a_silent_backend() -> None:
    """`retry_on` is a public parameter. A timeout produced no answer, so there
    is nothing for the caller to act on and the chain must still advance."""
    chain = FallbackLLM(
        [_Backend("slow", delay=3.0), _Backend("fast")],
        retry_on=(ConnectionError,),
        timeout_s=0.2,
    )
    assert chain.invoke(MESSAGES) == "fast answer"


async def test_the_async_path_advances_under_a_narrowed_retry_on_too() -> None:
    chain = FallbackLLM(
        [_Backend("slow", delay=3.0), _Backend("fast")],
        retry_on=(ConnectionError,),
        timeout_s=0.2,
    )
    assert await chain.ainvoke(MESSAGES) == "fast answer"


class _Timeouts:
    """Raises `TimeoutError` itself, rather than being slow."""

    backend_name = "timeouts"
    backend_version = "0"

    def invoke(self, messages: list[Message], **kwargs: Any) -> str:
        raise TimeoutError("the backend's own timeout")

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> str:
        raise TimeoutError("the backend's own timeout")

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("timeout_s", [None, 5.0])
async def test_a_backend_raised_timeout_is_routed_the_same_way_on_both_paths(
    use_async: bool, timeout_s: float | None
) -> None:
    """A `TimeoutError` the backend raises is the backend's, not the deadline's,
    so it goes through `retry_on`. Both paths must agree, and the case that
    matters is with a deadline set. Without one the async branch never runs."""
    chain = FallbackLLM(
        [_Timeouts(), _Backend("fast")],
        retry_on=(ConnectionError,),
        timeout_s=timeout_s,
    )
    with pytest.raises(TimeoutError):
        await chain.ainvoke(MESSAGES) if use_async else chain.invoke(MESSAGES)


def test_a_budget_trip_raised_after_the_deadline_still_advances_the_chain() -> None:
    """Otherwise an exhausted provider is billed again on every later call."""

    class SlowBudget:
        backend_name = "slow-budget"
        backend_version = "0"

        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list[Message], **kwargs: Any) -> str:
            self.calls += 1
            time.sleep(0.5)
            raise BudgetExhausted("ceiling hit")

        async def ainvoke(self, messages: list[Message], **kwargs: Any) -> str:
            raise BudgetExhausted("ceiling hit")

        def close(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    exhausted = SlowBudget()
    chain = FallbackLLM([exhausted, _Backend("fast")], timeout_s=0.1)
    for _ in range(3):
        assert chain.invoke(MESSAGES) == "fast answer"
    time.sleep(0.8)  # let the abandoned worker land its BudgetExhausted
    assert chain.invoke(MESSAGES) == "fast answer"
    assert chain.active_index == 1, "the exhausted backend was never skipped"
    assert exhausted.calls <= 3, f"billed {exhausted.calls} times after exhaustion"


def test_the_high_water_mark_only_ever_moves_forward() -> None:
    """A timed-out worker trips it from its own thread carrying a stale index."""
    chain = FallbackLLM([_Backend("a"), _Backend("b"), _Backend("c")])
    chain._trip_past(1)
    assert chain.active_index == 2
    chain._trip_past(0)
    assert chain.active_index == 2


def test_the_deadline_applies_past_the_first_position() -> None:
    """Once `_active` has advanced, the deadline must still hold."""
    chain = FallbackLLM(
        [_Backend("a"), _Backend("slow", delay=3.0), _Backend("fast")], timeout_s=0.2
    )
    chain._active = 1
    start = time.perf_counter()
    assert chain.invoke(MESSAGES) == "fast answer"
    assert time.perf_counter() - start < 1.5


def test_the_abandoned_worker_cannot_hold_interpreter_shutdown() -> None:
    """A non-daemon worker keeps the process alive until the provider returns."""
    script = (
        "import time,sys;"
        "from rag_llm_infra.fallback import FallbackLLM\n"
        "class B:\n"
        "    backend_name='b'\n"
        "    def __init__(self,d): self.d=d\n"
        "    def invoke(self,m,**k):\n"
        "        time.sleep(self.d); return 'slow'\n"
        "    async def ainvoke(self,m,**k): return 'slow'\n"
        "class F(B):\n"
        "    def invoke(self,m,**k): return 'fast'\n"
        "print(FallbackLLM([B(20.0),F(0)],timeout_s=0.1).invoke([]))"
    )
    start = time.perf_counter()
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    elapsed = time.perf_counter() - start
    assert done.returncode == 0, done.stderr[-500:]
    assert elapsed < 10, f"process took {elapsed:.1f}s; the worker held shutdown"
