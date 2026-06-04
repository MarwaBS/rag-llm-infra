# RAG + LLM Serving Infrastructure

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

A small, vendor-neutral foundation for retrieval-augmented LLM applications:
a swappable vector store, a cached embedding index, a provider-agnostic LLM
protocol, and the observability around them.

> Distilled infrastructure layer, extracted from a larger private application —
> typed, tested, and runnable on its own. The application that consumed it is
> not part of this repository.

## Quickstart — end-to-end RAG (no API key, no network)

```bash
pip install numpy
python example.py
```

[`example.py`](example.py) wires the three pieces into one flow:

```
embed documents → index in a VectorStore → retrieve top-k for a query
                → build a grounded prompt → answer with an LLMProtocol backend
```

It runs on the NumPy vector store and the deterministic mock LLM, so it needs
no key and no native libraries. In production, swap `embed()` for
`EmbeddingEngine` (real sentence embeddings) and `get_llm("mock")` for
`get_llm("openai")`.

## What's inside

| Module | Responsibility |
| --- | --- |
| `llm_protocol.py` | `LLMProtocol` — a `runtime_checkable` Protocol over **OpenAI**, an **Anthropic** contract-stub, and a deterministic **Mock** backend, so the model vendor is a config choice. Factory: `get_llm(backend)`. |
| `vector_store.py` | `VectorStoreProtocol` with **three** backends — in-process FAISS `IndexFlatIP`, a pure-NumPy `argpartition` fallback, and a real **Qdrant** backend (embedded `:memory:` or managed via `QDRANT_URL`) with batched search. |
| `evidence_index.py` | `EmbeddingEngine` — SentenceTransformers embeddings behind an adaptive, memory-pressure-aware LRU cache; plus a reader/writer lock for concurrent read / exclusive write. |
| `tracing.py` | OpenTelemetry spans with a console-exporter fallback when no OTLP endpoint is set, and a no-op tracer when OTel isn't installed. |
| `log_config.py` | Structured JSON logging with trace-id injection and an `llm_call` latency/token timing context manager. |

## Engineering principles demonstrated

- **Swap by interface** — `LLMProtocol` and `VectorStoreProtocol` make the model
  and the index runtime-swappable without touching call sites.
- **Degrade, don't crash** — optional dependencies (FAISS, Qdrant, OpenTelemetry,
  SentenceTransformers) are lazily imported with working fallbacks; missing
  infrastructure never hard-fails module import.
- **Typed and tested** — `from __future__ import annotations`, `Protocol` /
  `runtime_checkable` boundaries, and hermetic unit tests built on a deterministic
  mock backend (no network calls).

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the end-to-end RAG pipeline, the LLM protocol (conformance +
factory routing), the vector store (known-answer search across NumPy/FAISS/Qdrant),
the embedding index and reader/writer lock, tracing, and structured logging — all
hermetic.

## License

MIT — see [LICENSE](LICENSE).
