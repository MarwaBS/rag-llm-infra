"""
LLMProtocol — vendor-neutral chat-completion abstraction.

Mirrors `vector_store.py:VectorStoreProtocol` for LLM calls. See
`docs/decisions/006-llm-protocol-abstraction.md` for the full rationale.

Runtime backends:
    - OpenAIBackend  — production default
    - AnthropicBackend — contract stub, raises NotImplementedError with
      migration TODO. Locks the interface so a later implementer does not
      have to redesign call sites.
    - MockBackend    — deterministic, no network, no cost; for tests.

Selection is via `get_llm(backend)` reading an `auto | openai | anthropic
| mock` value.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

__all__ = [
    "LLMProtocol",
    "OpenAIBackend",
    "AnthropicBackend",
    "MockBackend",
    "get_llm",
]


@runtime_checkable
class LLMProtocol(Protocol):
    """Minimal chat-completion contract used by callers.

    Implementations wrap a specific LLM vendor behind a uniform surface so
    callers can switch backends via config. This protocol deliberately does
    NOT include streaming, tool-calling, or vision inputs — those are
    vendor-specific today and would leak the abstraction. Narrow beats
    feature-complete; add surface when a second real backend proves the
    need.

    Cost tracking is deliberately NOT handled here — it belongs at the
    service-layer boundary, so every backend benefits from one cost
    ceiling without reimplementing it per vendor.
    """

    backend_name: str
    backend_version: str

    def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Synchronous chat completion. Returns the assistant text."""
        ...

    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Async chat completion. Returns the assistant text."""
        ...


class OpenAIBackend:
    """Wraps `openai.OpenAI` / `openai.AsyncOpenAI`.

    `model` and `api_key` are pass-through defaults, not recommendations: no
    model was benchmarked here, and `api_key=None` falls through to the SDK's
    own `OPENAI_API_KEY` lookup. Pass `model=` to pin the one you have measured.
    """

    backend_name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        try:
            import openai  # lazy: importing this module must not need the SDK
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIBackend requires the `openai` package. "
                "Install `openai>=1.0` or pick another backend via get_llm(backend=...)."
            ) from exc

        # Construct the SDK clients lazily: building both eagerly would open an
        # httpx pool a purely sync caller never touches and never closes.
        self._openai = openai
        self._api_key = api_key
        self._model = model
        self._client: Any | None = None
        self._aclient: Any | None = None
        self.backend_version = openai.__version__

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._openai.OpenAI(api_key=self._api_key)
        return self._client

    @property
    def aclient(self) -> Any:
        if self._aclient is None:
            self._aclient = self._openai.AsyncOpenAI(api_key=self._api_key)
        return self._aclient

    def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        # openai's SDK uses a typed-dict union for ChatCompletionMessageParam;
        # at runtime it accepts any dict shape with 'role' + 'content'. Our
        # Protocol surface is intentionally the simpler shape, so we cast.
        resp = self.client.chat.completions.create(
            model=self._model,
            messages=cast(Any, messages),
            **kwargs,
        )
        return cast(str, resp.choices[0].message.content or "")

    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        resp = await self.aclient.chat.completions.create(
            model=self._model,
            messages=cast(Any, messages),
            **kwargs,
        )
        return cast(str, resp.choices[0].message.content or "")

    def close(self) -> None:
        """Close the sync client if one was built. The async client needs
        `aclose()`: `AsyncOpenAI.close` is a coroutine function, so calling it
        from here would discard the coroutine and close nothing."""
        if self._client is not None:
            self._client.close()

    async def aclose(self) -> None:
        """Close the async client if one was built."""
        if self._aclient is not None:
            await self._aclient.close()


class AnthropicBackend:
    """Intentionally unimplemented. Exists to lock the protocol surface.

    Migration path when promoting to a real backend:

    1. `pip install anthropic` (add it to the project dependencies in pyproject.toml).
    2. Replace both `invoke` / `ainvoke` bodies with calls to
       `anthropic.Anthropic().messages.create(model=..., system=..., messages=...)`
       and `anthropic.AsyncAnthropic()` respectively.
    3. Anthropic expects `system` as a top-level arg, not a role in
       `messages`. Extract the system message before the API call.
    4. Normalize the response: `resp.content[0].text` -> `str`.
    5. Map OpenAI-style kwargs (`temperature`, `max_tokens`) 1:1.
    6. Update `backend_version = anthropic.__version__`.
    7. Add a live-call test gated on `ANTHROPIC_API_KEY` being present.
    """

    backend_name = "anthropic"
    backend_version = "stub"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key

    def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        raise NotImplementedError(
            "AnthropicBackend is a contract stub. Implement per the docstring "
            "in llm_protocol.py before use. See "
            "docs/decisions/006-llm-protocol-abstraction.md for the full plan."
        )

    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        raise NotImplementedError(
            "AnthropicBackend is a contract stub. Implement per the docstring "
            "in llm_protocol.py before use. See "
            "docs/decisions/006-llm-protocol-abstraction.md for the full plan."
        )


class MockBackend:
    """Deterministic backend for tests and local dry-runs. No network, no cost.

    `response` can be a plain string (always returned) or a callable taking
    the messages list and returning a string (lets tests assert on input
    shape).
    """

    backend_name = "mock"
    backend_version = "1.0.0"

    def __init__(self, response: Any = "MOCK_RESPONSE") -> None:
        self._response = response

    def _resolve(self, messages: list[dict[str, str]]) -> str:
        if callable(self._response):
            return str(self._response(messages))
        return str(self._response)

    def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return self._resolve(messages)

    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return self._resolve(messages)


_VALID_BACKENDS = ("auto", "openai", "anthropic", "mock")


def get_llm(backend: str = "auto", **kwargs: Any) -> LLMProtocol:
    """Select an `LLMProtocol` implementation by name.

    `backend`:
        - `auto`      — an alias for `openai`, not a capability probe. Raises
                        when `openai` is absent rather than degrading to
                        `mock`: a fake answer is worse than no answer. Unlike
                        `get_vector_store("auto")`, where every backend
                        returns real neighbours and falling back is safe.
        - `openai`    — explicit OpenAI.
        - `anthropic` — stub; raises at call time (see AnthropicBackend).
        - `mock`      — deterministic, for tests.

    Extra kwargs are forwarded to the backend constructor.
    """
    name = backend.lower()
    if name not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown llm backend: {backend!r}. Valid: {' | '.join(_VALID_BACKENDS)}."
        )

    if name in ("auto", "openai"):
        return OpenAIBackend(**kwargs)
    if name == "anthropic":
        return AnthropicBackend(**kwargs)
    if name == "mock":
        return MockBackend(**kwargs)

    # Unreachable given the validation above; keeps mypy happy.
    raise ValueError(f"Unreachable: backend={backend!r}")
