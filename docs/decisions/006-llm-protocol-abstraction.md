# ADR-006: Vendor-neutral LLM protocol abstraction

- **Status:** Accepted
- **Context area:** `src/rag_llm_infra/llm_protocol.py`, `src/rag_llm_infra/fallback.py`

## Context

The library makes chat-completion calls to an LLM vendor. Call sites that import a
vendor SDK directly (`openai.OpenAI().chat.completions.create(...)`) couple every
caller to one vendor's surface: switching providers, adding a fallback chain, or
running deterministic tests means editing each call site. We also want to add a
second provider (Anthropic) later without redesigning callers, and to test the
generation path with no network and no API key.

## Decision

Define a narrow `LLMProtocol` (`invoke` / `ainvoke`, both returning assistant
text) and select an implementation through a `get_llm(backend)` factory:

- `OpenAIBackend` — the production default; wraps `openai.OpenAI` / `AsyncOpenAI`.
- `AnthropicBackend` — a **contract stub** that raises `NotImplementedError` with a
  step-by-step migration plan in its docstring. It exists to lock the interface so
  a later implementer changes one class, not every call site. It is deliberately
  *not* silently skippable: a fallback chain must not include it and quietly mask
  the gap (see ADR consequences and `FallbackLLM`'s non-retryable error list).
- `MockBackend` — deterministic, no network, no cost; the default in tests and the
  reference service.

The protocol is intentionally minimal: no streaming, tool-calling, or vision
inputs. Those are vendor-specific today and would leak the abstraction. Cost
tracking lives at the service-layer boundary, not in each backend, so one cost
ceiling covers every vendor.

`FallbackLLM` composes backends behind the same protocol: it advances to the next
backend on a transient provider error and skips a `BudgetExhausted` provider
permanently. Programming/contract errors (e.g. `NotImplementedError`,
`TypeError`) are **not** retryable — they propagate, so a misconfigured chain
fails loudly instead of silently degrading.

## Consequences

- Switching vendors or adding a fallback is a config change, not a rewrite.
- The generation path is testable offline via `MockBackend`.
- A new real backend implements two methods and is wired into `get_llm`.
- The stub's loud failure is intentional: chaining an unimplemented backend is a
  bug, not a runtime fallback path.
