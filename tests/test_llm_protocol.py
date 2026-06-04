"""
LLMProtocol conformance + factory tests.

Each backend is verified against the Protocol via isinstance(LLMProtocol),
plus per-backend behaviors (mock determinism, anthropic stub error message,
factory routing).

No live API calls — OpenAIBackend's network path is NOT exercised here.
These tests are hermetic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_protocol import (
    AnthropicBackend,
    LLMProtocol,
    MockBackend,
    OpenAIBackend,
    get_llm,
)


# ---------------------------------------------------------------------------
# MockBackend — the workhorse for every other test in the repo that needs
# a deterministic LLM. Must be boring and reliable.
# ---------------------------------------------------------------------------
class TestMockBackend:
    def test_default_response(self) -> None:
        llm = MockBackend()
        assert llm.invoke([{"role": "user", "content": "hi"}]) == "MOCK_RESPONSE"

    def test_custom_string_response(self) -> None:
        llm = MockBackend(response="custom")
        assert llm.invoke([{"role": "user", "content": "?"}]) == "custom"

    def test_callable_response_receives_messages(self) -> None:
        captured: list = []

        def echo(messages):
            captured.append(messages)
            return f"seen:{len(messages)}"

        llm = MockBackend(response=echo)
        result = llm.invoke([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
        assert result == "seen:2"
        assert len(captured) == 1
        assert captured[0][0]["content"] == "a"

    @pytest.mark.asyncio
    async def test_async_returns_same_as_sync(self) -> None:
        llm = MockBackend(response="both-paths")
        assert llm.invoke([]) == "both-paths"
        assert await llm.ainvoke([]) == "both-paths"

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(MockBackend(), LLMProtocol)

    def test_has_backend_metadata(self) -> None:
        llm = MockBackend()
        assert llm.backend_name == "mock"
        assert llm.backend_version  # non-empty

    def test_kwargs_ignored_silently(self) -> None:
        """Mock ignores temperature/max_tokens/etc — caller doesn't need to
        strip them when swapping in MockBackend for tests."""
        llm = MockBackend(response="ok")
        assert llm.invoke([], temperature=0.7, max_tokens=100) == "ok"


# ---------------------------------------------------------------------------
# AnthropicBackend — contract stub. The only wrong thing it can do is
# silently succeed, so we test that it raises loudly with migration help.
# ---------------------------------------------------------------------------
class TestAnthropicBackend:
    def test_invoke_raises_not_implemented(self) -> None:
        llm = AnthropicBackend()
        with pytest.raises(NotImplementedError, match="contract stub"):
            llm.invoke([{"role": "user", "content": "?"}])

    @pytest.mark.asyncio
    async def test_ainvoke_raises_not_implemented(self) -> None:
        llm = AnthropicBackend()
        with pytest.raises(NotImplementedError, match="contract stub"):
            await llm.ainvoke([{"role": "user", "content": "?"}])

    def test_error_points_at_adr(self) -> None:
        """Failure message must include the ADR path so an implementer knows
        where the migration plan lives."""
        llm = AnthropicBackend()
        with pytest.raises(NotImplementedError, match="006-llm-protocol-abstraction"):
            llm.invoke([])

    def test_conforms_to_protocol_even_as_stub(self) -> None:
        """runtime_checkable Protocol verifies shape only — the stub has the
        right methods, so isinstance passes. NotImplementedError fires at
        call time, which is the point of the stub."""
        assert isinstance(AnthropicBackend(), LLMProtocol)

    def test_defaults_to_claude_sonnet(self) -> None:
        llm = AnthropicBackend()
        assert llm._model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# OpenAIBackend — import-time tests only (no network), via a patched SDK.
# ---------------------------------------------------------------------------
class TestOpenAIBackend:
    def test_instantiation_with_patched_sdk(self) -> None:
        """Construct with a mocked openai module so the test is hermetic."""
        fake_openai = MagicMock()
        fake_openai.__version__ = "1.109.1"
        with patch.dict("sys.modules", {"openai": fake_openai}):
            llm = OpenAIBackend(model="gpt-4o-mini")
            assert llm.backend_name == "openai"
            assert llm.backend_version == "1.109.1"
            assert llm._model == "gpt-4o-mini"

    def test_conforms_to_protocol(self) -> None:
        fake_openai = MagicMock()
        fake_openai.__version__ = "1.109.1"
        with patch.dict("sys.modules", {"openai": fake_openai}):
            llm = OpenAIBackend()
            assert isinstance(llm, LLMProtocol)

    def test_missing_sdk_raises_runtime_error(self) -> None:
        """Simulate `openai` not installed — constructor must raise with
        install guidance, not a cryptic ImportError from deep in the call."""
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(RuntimeError, match="openai"):
                OpenAIBackend()


# ---------------------------------------------------------------------------
# Factory — get_llm routing is the only place call sites touch, so every
# routing branch needs a test.
# ---------------------------------------------------------------------------
class TestFactory:
    def test_mock_routing(self) -> None:
        llm = get_llm(backend="mock")
        assert llm.backend_name == "mock"

    def test_anthropic_routing_returns_stub(self) -> None:
        llm = get_llm(backend="anthropic")
        assert llm.backend_name == "anthropic"
        with pytest.raises(NotImplementedError):
            llm.invoke([])

    def test_auto_routes_to_openai(self) -> None:
        """`auto` must equal `openai` in production. Verified by patching
        OpenAIBackend to a sentinel and asserting it's constructed."""
        sentinel = MockBackend(response="sentinel")
        with patch("llm_protocol.OpenAIBackend", return_value=sentinel) as ctor:
            llm = get_llm(backend="auto")
            assert ctor.called
            assert llm is sentinel

    def test_openai_explicit_routes_to_openai(self) -> None:
        sentinel = MockBackend(response="sentinel")
        with patch("llm_protocol.OpenAIBackend", return_value=sentinel) as ctor:
            llm = get_llm(backend="openai")
            assert ctor.called
            assert llm is sentinel

    def test_case_insensitive(self) -> None:
        assert get_llm(backend="MOCK").backend_name == "mock"
        assert get_llm(backend="Mock").backend_name == "mock"

    def test_unknown_backend_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown llm backend"):
            get_llm(backend="grok")

    def test_unknown_backend_error_lists_valid_options(self) -> None:
        """The error must help the caller fix it — list the valid names."""
        with pytest.raises(ValueError) as exc:
            get_llm(backend="palm")
        assert "openai" in str(exc.value)
        assert "anthropic" in str(exc.value)
        assert "mock" in str(exc.value)

    def test_kwargs_forwarded_to_backend(self) -> None:
        llm = get_llm(backend="mock", response="forwarded")
        assert llm.invoke([]) == "forwarded"


# ---------------------------------------------------------------------------
# Protocol shape — if someone adds a field / method, this test flags the
# drift instantly.
# ---------------------------------------------------------------------------
class TestProtocolShape:
    @pytest.mark.parametrize(
        "impl_cls",
        [MockBackend, AnthropicBackend],
    )
    def test_backend_has_required_fields(self, impl_cls) -> None:
        instance = impl_cls()
        assert hasattr(instance, "backend_name")
        assert hasattr(instance, "backend_version")
        assert hasattr(instance, "invoke")
        assert hasattr(instance, "ainvoke")

    @pytest.mark.parametrize(
        "impl_cls",
        [MockBackend, AnthropicBackend],
    )
    def test_backend_name_is_nonempty_string(self, impl_cls) -> None:
        instance = impl_cls()
        assert isinstance(instance.backend_name, str) and instance.backend_name
