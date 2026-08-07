"""Multi-provider LLM fallback with a permanent budget-exhaustion trip.

`FallbackLLM` wraps an ordered list of `LLMProtocol` backends and advances to the
next one when the current backend raises a *retryable* error. It does NOT track
spend itself. Budget accounting lives at the service layer (see ADR-006). This
class only *reacts* to a `BudgetExhausted` a backend raises, and trips the chain
forward **permanently** — that provider is skipped for the rest of this object's
life. Other retryable exceptions are transient: the next backend is tried for
that call only.

A provider that hangs raises nothing, so without a deadline it holds the whole
chain. Pass `timeout_s=` and a backend that does not answer in time raises
`BackendTimeout`. That is handled ahead of `retry_on`, so narrowing `retry_on`
cannot leave the chain stuck on a silent backend.

Bounded: on the sync path this stops waiting, it does not cancel. The provider
keeps working on a daemon thread and its answer is discarded. Python cannot
interrupt a blocking socket read in another thread. What is not discarded is a
`BudgetExhausted` it raises afterwards — that still trips the chain past it.
The async path uses `asyncio.wait_for`, which does cancel. Without `timeout_s`
the behaviour is unchanged and a hanging provider blocks.

Programming/contract errors (e.g. `TypeError`, `NotImplementedError`) are NOT
retryable — they propagate, so a misconfigured chain fails loudly instead of
silently degrading. That is also why you should not chain the `AnthropicBackend`
stub: it raises `NotImplementedError`, which is a bug to surface, not a fallback.

Conforms to `LLMProtocol`, so it is a drop-in anywhere a single backend is used::

    from rag_llm_infra import get_llm, FallbackLLM
    llm = FallbackLLM([get_llm("openai"), get_llm("mock")])

Thread safety: a single `FallbackLLM` is safe to share across threads. The only
mutable state is `_active`, the budget-exhaustion high-water mark. Its
read-modify-write is taken under a lock, so it advances monotonically even when
a timed-out worker writes it from its own thread. The backends do the real work
outside that lock, so there is no added contention.

Bounded: two threads racing on the same just-exhausted backend may each see the
`BudgetExhausted` once before `_active` settles. The chain still trips forward.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from .llm_protocol import LLMProtocol, Message

# Errors that signal a bug or contract violation, not a recoverable provider
# failure. These always propagate, even if `retry_on` would otherwise match them,
# so fallback never masks a programming error.
_NON_RETRYABLE: tuple[type[BaseException], ...] = (
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    NameError,
    NotImplementedError,
)


class BudgetExhausted(RuntimeError):
    """Raised by a backend when its spend ceiling is hit. Trips `FallbackLLM`
    forward permanently rather than retrying the exhausted provider."""


class BackendTimeout(TimeoutError):
    """A backend did not answer inside `timeout_s`.

    Handled ahead of `retry_on`, so narrowing that parameter cannot stop the
    chain advancing past a backend that produced nothing.
    """


class FallbackLLM:
    """Route to the next backend on failure; conforms to `LLMProtocol`."""

    backend_name = "fallback"

    def __init__(
        self,
        backends: Sequence[LLMProtocol],
        *,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
        timeout_s: float | None = None,
    ) -> None:
        self._backends: list[LLMProtocol] = list(backends)
        if not self._backends:
            raise ValueError("FallbackLLM requires at least one backend")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        self._retry_on = retry_on
        self._timeout_s = timeout_s
        self._active = 0
        # A timed-out worker writes `_active` from its own thread.
        self._active_lock = threading.Lock()
        self.backend_version = "+".join(b.backend_name for b in self._backends)

    def _trip_past(self, index: int) -> None:
        """Skip backend `index` for the rest of this object's life."""
        with self._active_lock:
            self._active = max(self._active, index + 1)

    def _within_deadline(self, index: int, call: Callable[[], str]) -> str:
        """Run `call`, giving up on it after `timeout_s`.

        The worker is a daemon so an abandoned call cannot hold interpreter
        shutdown. Giving up is not cancelling: the provider keeps working and its
        answer is discarded. Python cannot interrupt a blocking socket read in
        another thread.

        An abandoned call that later raises `BudgetExhausted` still trips the
        chain past that backend. Dropping it would let an exhausted provider be
        billed again on every subsequent call.
        """
        if self._timeout_s is None:
            return call()
        answer: list[str] = []
        failure: list[BaseException] = []

        def run() -> None:
            try:
                answer.append(call())
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                failure.append(exc)
                if isinstance(exc, BudgetExhausted):
                    self._trip_past(index)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(self._timeout_s)
        if worker.is_alive():
            raise BackendTimeout(f"backend did not answer within {self._timeout_s}s")
        if failure:
            raise failure[0]
        return answer[0]

    def close(self) -> None:
        """Close every backend in the chain, including ones already tripped."""
        for backend in self._backends:
            backend.close()

    async def aclose(self) -> None:
        for backend in self._backends:
            await backend.aclose()

    @property
    def active_index(self) -> int:
        """Index of the first backend still eligible (advances past exhausted ones)."""
        return self._active

    def invoke(self, messages: list[Message], **kwargs: Any) -> str:
        last: BaseException | None = None
        for i in range(self._active, len(self._backends)):
            try:
                backend = self._backends[i]
                return self._within_deadline(
                    i, partial(backend.invoke, messages, **kwargs)
                )
            except BackendTimeout as exc:
                # Always retryable, whatever `retry_on` narrows to: no answer
                # arrived, so there is nothing for the caller to act on.
                last = exc
            except BudgetExhausted as exc:
                last = exc
                # Permanent: never retry an exhausted backend. `max` keeps the
                # advance monotonic so a slower concurrent call can't regress it.
                self._trip_past(i)
            except _NON_RETRYABLE:
                raise  # a bug, not a provider failure — surface it, don't fall through
            except self._retry_on as exc:
                last = exc  # transient: try the next backend for this call only
        raise RuntimeError(
            f"FallbackLLM: all {len(self._backends)} backends failed"
        ) from last

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> str:
        last: BaseException | None = None
        for i in range(self._active, len(self._backends)):
            try:
                call = self._backends[i].ainvoke(messages, **kwargs)
                if self._timeout_s is None:
                    return await call
                task = asyncio.ensure_future(call)
                try:
                    return await asyncio.wait_for(task, self._timeout_s)
                except TimeoutError as exc:
                    # Cancellation tells our deadline from the backend's own
                    # TimeoutError; the type cannot.
                    if not task.cancelled():
                        raise
                    raise BackendTimeout(
                        f"backend did not answer within {self._timeout_s}s"
                    ) from exc
            except BackendTimeout as exc:
                last = exc
            except BudgetExhausted as exc:
                last = exc
                self._trip_past(i)
            except _NON_RETRYABLE:
                raise
            except self._retry_on as exc:
                last = exc
        raise RuntimeError(
            f"FallbackLLM: all {len(self._backends)} backends failed"
        ) from last
