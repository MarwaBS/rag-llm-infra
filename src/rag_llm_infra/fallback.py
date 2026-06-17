"""Multi-provider LLM fallback with a permanent budget-exhaustion trip.

`FallbackLLM` wraps an ordered list of `LLMProtocol` backends and advances to the
next one when the current backend raises a *retryable* error. It does NOT track
spend itself — budget accounting lives at the service layer (see ADR-006); this
class only *reacts* to a `BudgetExhausted` signal a backend raises, by tripping
the chain forward **permanently** (the exhausted provider is skipped for the rest
of this object's life). Other retryable exceptions are transient: the next
backend is tried for that call only.

Programming/contract errors (e.g. `TypeError`, `NotImplementedError`) are NOT
retryable — they propagate, so a misconfigured chain fails loudly instead of
silently degrading. That is also why you should not chain the `AnthropicBackend`
stub: it raises `NotImplementedError`, which is a bug to surface, not a fallback.

Conforms to `LLMProtocol`, so it is a drop-in anywhere a single backend is used::

    from rag_llm_infra import get_llm, FallbackLLM
    llm = FallbackLLM([get_llm("openai"), get_llm("mock")])

Thread safety: a single `FallbackLLM` is safe to share across threads. The only
mutable state is `_active` (the budget-exhaustion high-water mark), which advances
*monotonically* — concurrent calls can never move it backward, and the underlying
backends do the real work outside any lock, so there is no added contention. The
benign cost of being lock-free is that two threads racing on the same just-exhausted
backend may each discover the `BudgetExhausted` once before `_active` settles; the
chain still trips forward correctly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .llm_protocol import LLMProtocol

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


class FallbackLLM:
    """Route to the next backend on failure; conforms to `LLMProtocol`."""

    backend_name = "fallback"

    def __init__(
        self,
        backends: Sequence[LLMProtocol],
        *,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        self._backends: list[LLMProtocol] = list(backends)
        if not self._backends:
            raise ValueError("FallbackLLM requires at least one backend")
        self._retry_on = retry_on
        self._active = 0
        self.backend_version = "+".join(b.backend_name for b in self._backends)

    @property
    def active_index(self) -> int:
        """Index of the first backend still eligible (advances past exhausted ones)."""
        return self._active

    def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        last: BaseException | None = None
        for i in range(self._active, len(self._backends)):
            try:
                return self._backends[i].invoke(messages, **kwargs)
            except BudgetExhausted as exc:
                last = exc
                # Permanent: never retry an exhausted backend. `max` keeps the
                # advance monotonic so a slower concurrent call can't regress it.
                self._active = max(self._active, i + 1)
            except _NON_RETRYABLE:
                raise  # a bug, not a provider failure — surface it, don't fall through
            except self._retry_on as exc:
                last = exc  # transient: try the next backend for this call only
        raise RuntimeError(
            f"FallbackLLM: all {len(self._backends)} backends failed"
        ) from last

    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        last: BaseException | None = None
        for i in range(self._active, len(self._backends)):
            try:
                return await self._backends[i].ainvoke(messages, **kwargs)
            except BudgetExhausted as exc:
                last = exc
                self._active = max(self._active, i + 1)
            except _NON_RETRYABLE:
                raise
            except self._retry_on as exc:
                last = exc
        raise RuntimeError(
            f"FallbackLLM: all {len(self._backends)} backends failed"
        ) from last
