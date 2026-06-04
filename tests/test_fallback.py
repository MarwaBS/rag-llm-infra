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
