"""Tests for FallbackLLM — transient fallthrough vs permanent budget trip."""

import pytest

from rag_llm_infra import BudgetExhausted, FallbackLLM, LLMProtocol, MockBackend


class _Boom:
    """Backend that always raises the given exception. Conforms to LLMProtocol."""

    backend_name = "boom"
    backend_version = "0"

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def invoke(self, messages, **kwargs):
        raise self._exc

    async def ainvoke(self, messages, **kwargs):
        raise self._exc


def test_conforms_to_protocol() -> None:
    assert isinstance(FallbackLLM([MockBackend()]), LLMProtocol)


def test_requires_at_least_one_backend() -> None:
    with pytest.raises(ValueError):
        FallbackLLM([])


def test_falls_through_transient_failure() -> None:
    llm = FallbackLLM([_Boom(RuntimeError("rate limit")), MockBackend(response="ok")])
    assert llm.invoke([]) == "ok"


def test_all_backends_failing_raises() -> None:
    llm = FallbackLLM([_Boom(RuntimeError("a")), _Boom(RuntimeError("b"))])
    with pytest.raises(RuntimeError, match="all 2 backends failed"):
        llm.invoke([])


def test_budget_exhausted_advances_permanently() -> None:
    primary = _Boom(BudgetExhausted("daily ceiling hit"))
    llm = FallbackLLM([primary, MockBackend(response="secondary")])
    assert llm.active_index == 0
    assert llm.invoke([]) == "secondary"
    # The exhausted primary is skipped permanently on the next call.
    assert llm.active_index == 1
    assert llm.invoke([]) == "secondary"


def test_transient_failure_does_not_advance_permanently() -> None:
    llm = FallbackLLM([_Boom(RuntimeError("blip")), MockBackend(response="ok")])
    llm.invoke([])
    assert llm.active_index == 0  # transient errors do not burn the primary


@pytest.mark.asyncio
async def test_async_fallthrough() -> None:
    llm = FallbackLLM([_Boom(RuntimeError("x")), MockBackend(response="async-ok")])
    assert await llm.ainvoke([]) == "async-ok"


def test_programming_error_propagates_not_masked() -> None:
    """A TypeError is a bug, not a provider failure. It must NOT fall through to
    the next backend (which would hide the bug behind a 'working' response)."""
    llm = FallbackLLM(
        [_Boom(TypeError("bad argument")), MockBackend(response="masked")]
    )
    with pytest.raises(TypeError, match="bad argument"):
        llm.invoke([])


def test_not_implemented_stub_propagates() -> None:
    """Chaining the AnthropicBackend stub (NotImplementedError) must surface the
    misconfiguration, not silently skip to the next backend."""
    llm = FallbackLLM(
        [_Boom(NotImplementedError("stub")), MockBackend(response="masked")]
    )
    with pytest.raises(NotImplementedError):
        llm.invoke([])


@pytest.mark.asyncio
async def test_async_programming_error_propagates() -> None:
    llm = FallbackLLM([_Boom(KeyError("missing")), MockBackend(response="masked")])
    with pytest.raises(KeyError):
        await llm.ainvoke([])


@pytest.mark.asyncio
async def test_async_budget_exhausted_advances_permanently() -> None:
    """The trip must be permanent on the async path too, not just the sync one."""
    primary = _Boom(BudgetExhausted("daily ceiling hit"))
    llm = FallbackLLM([primary, MockBackend(response="secondary")])
    assert await llm.ainvoke([]) == "secondary"
    assert llm.active_index == 1
    assert await llm.ainvoke([]) == "secondary"


@pytest.mark.asyncio
async def test_async_all_backends_failing_raises_with_the_last_cause() -> None:
    llm = FallbackLLM([_Boom(RuntimeError("first")), _Boom(RuntimeError("last"))])
    with pytest.raises(RuntimeError, match="all 2 backends failed") as exc:
        await llm.ainvoke([])
    assert str(exc.value.__cause__) == "last"


class _Counting:
    """Backend that records how many times it was asked."""

    backend_name = "counting"
    backend_version = "0"

    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        if self._exc:
            raise self._exc
        return "ok"

    async def ainvoke(self, messages, **kwargs):
        return self.invoke(messages, **kwargs)


def test_an_exhausted_backend_is_never_asked_again() -> None:
    """`active_index` advancing is the mechanism; not calling the dead backend
    again is the behaviour. Only the second one survives a rewrite of the first."""
    primary = _Counting(BudgetExhausted("ceiling"))
    llm = FallbackLLM([primary, MockBackend(response="secondary")])
    for _ in range(5):
        llm.invoke([])
    assert primary.calls == 1


@pytest.mark.asyncio
async def test_an_exhausted_backend_is_never_awaited_again() -> None:
    primary = _Counting(BudgetExhausted("ceiling"))
    llm = FallbackLLM([primary, MockBackend(response="secondary")])
    for _ in range(5):
        await llm.ainvoke([])
    assert primary.calls == 1


def test_any_provider_exception_falls_through_by_default() -> None:
    """`retry_on` defaults to Exception so an unforeseen provider error still
    reaches the next backend. Narrowing it turns one into a hard failure."""
    llm = FallbackLLM(
        [_Boom(ValueError("provider said no")), MockBackend(response="ok")]
    )
    assert llm.invoke([]) == "ok"
