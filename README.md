# RAG + LLM Serving Infrastructure

[![CI](https://github.com/MarwaBS/rag-llm-infra/actions/workflows/ci.yml/badge.svg)](https://github.com/MarwaBS/rag-llm-infra/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rag-llm-infra)](https://pypi.org/project/rag-llm-infra/)

An installable, vendor-neutral foundation for retrieval-augmented LLM applications:
a swappable vector store, a cached embedding index, a provider-agnostic LLM
protocol, the observability around them, a FastAPI serving layer, and a
retrieval-quality eval gate.

> Typed, tested, packaged, and runnable on its own.

## Install

```bash
pip install rag-llm-infra                                   # core (numpy)
pip install "rag-llm-infra[faiss,qdrant,openai,serve]"      # + native backends, OpenAI, serving
pip install "rag-llm-infra[psutil]"                         # + memory-pressure-aware cache trimming
pip install -e ".[dev]"                                     # from a local clone, for development
```

## Quickstart: end-to-end RAG (no API key, no network)

```bash
git clone https://github.com/MarwaBS/rag-llm-infra && cd rag-llm-infra
pip install -e .
python example.py
```

```
embed documents -> index in a VectorStore -> retrieve top-k for a query
                -> build a grounded prompt -> answer with an LLMProtocol backend
```

Runs on the NumPy vector store + the deterministic mock LLM, so it needs no key.
In production, swap the demo embedder for `EmbeddingEngine` and `get_llm("mock")`
for `get_llm("openai")`.

## Serve it

```bash
pip install "rag-llm-infra[serve]"
export RAG_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
uvicorn rag_llm_infra.serve:app
# or: docker build -t rag-llm-infra . && docker run -p 8000:8000 -e RAG_API_KEY=$RAG_API_KEY rag-llm-infra
```

> `/index` and `/query` require `X-API-Key`, and answer 503 while `RAG_API_KEY`
> is unset. There is no open mode. `/health` stays open for container probes.
> Bodies over 1 MiB and corpora over 20000 documents are refused by default;
> each document costs a fixed-width vector however short it is, so bytes on the
> wire do not bound memory. `uvicorn` binds `127.0.0.1` unless `--host` or
> `UVICORN_HOST` says otherwise; the container passes `--host 0.0.0.0` and
> publishes 8000. One shared key, no rate limiting; see [SECURITY.md](SECURITY.md).

```bash
curl -XPOST localhost:8000/index -d '{"documents":["FAISS is in-process vector search","Qdrant is a vector database"]}' -H 'content-type: application/json' -H "X-API-Key: $RAG_API_KEY"
curl -XPOST localhost:8000/query -d '{"query":"vector search","k":1}'      -H 'content-type: application/json' -H "X-API-Key: $RAG_API_KEY"
```

## What's inside

| Module | Responsibility |
| --- | --- |
| `rag_llm_infra.llm_protocol` | `LLMProtocol`: `runtime_checkable` Protocol over OpenAI / Anthropic-stub / Mock; factory `get_llm()` |
| `rag_llm_infra.vector_store` | `VectorStoreProtocol`: in-process FAISS `IndexFlatIP`, pure-NumPy fallback, real **Qdrant** (batched search). Qdrant needs `collection=`: `add()` replaces that collection, so the store owns it |
| `rag_llm_infra.evidence_index` | `EmbeddingEngine`: SentenceTransformers embeddings + a cache (insertion-order eviction) guarded by a writer-preferring reader/writer lock, so the slow `model.encode` runs outside the lock. Memory-pressure-aware trimming activates with the `[psutil]` extra (`pip install "rag-llm-infra[psutil]"`); without it the cache is fixed-size |
| `rag_llm_infra.tracing` | OpenTelemetry spans with console-exporter + no-op fallbacks |
| `rag_llm_infra.log_config` | structured JSON logging + an `llm_call` timer. It measures latency; `tokens` is a field the caller fills |
| `rag_llm_infra.serve` | FastAPI service over the vector store + LLM protocol. `/index` and `/query` need `X-API-Key`; `/health`, `/docs`, `/redoc` and `/openapi.json` are open; see [SECURITY.md](SECURITY.md). Does not install `log_config` or `tracing`; call those yourself at startup |
| `rag_llm_infra.faithfulness` | `groundedness(answer, contexts)`: lexical faithfulness metric for RAG output |
| `rag_llm_infra.fallback` | `FallbackLLM`: budget-aware multi-provider routing; drop-in `LLMProtocol` |

## Quality gates

```bash
python -m eval.retrieval_eval      # recall@1 / MRR: retrieval mechanics over the demo embedder
python -m eval.generation_eval     # groundedness (faithfulness) of generated answers
```

Both run in CI: a **retrieval** regression or a **faithfulness** regression fails
the build and cannot merge. No floor is edited where it is used: every one is
computed by `scripts/derive_eval_floors.py` into `eval/eval_floors.json`, which
records the rule beside the measurement it came from, and a test requires
re-running the producer to reproduce that file byte for byte. The generation
floors come from the measured scores; the retrieval floors come from the query
count and a stated tolerance of one slipped rank.

`groundedness` is a **cheap lexical tripwire, not a faithfulness guarantee**. It
scores token overlap, so it has three blind spots by construction. It is
negation-blind: "X is not Y" reads as grounded. It is dilutable: a false clause
appended to a true answer only dents the score. And it scores vocabulary, not
propositions, so it cannot tell whether the evidence asserts the claim. It
catches the out-of-vocabulary hallucination signature cheaply on every
generation. Pair it with an LLM-judge for semantic faithfulness. The limits are
in the `faithfulness` module docstring and pinned by tests.

## Engineering principles shown

- **Swap by interface.** `LLMProtocol` / `VectorStoreProtocol` make the model and the index runtime-swappable.
- **Degrade, don't crash, where a degraded answer is still an answer.** FAISS / Qdrant / OpenTelemetry / SentenceTransformers are optional. Each is probed at import behind a handler that treats a missing library and one that fails to load alike, so neither stops `import rag_llm_infra`; a test simulates both. The LLM factory is the deliberate exception: `get_llm("auto")` raises rather than falling back to the mock backend, because a fabricated answer is worse than none.
- **Measured, not asserted.** A retrieval eval gate, not just unit tests; packaged and CI-built end to end.

## Develop / test

```bash
pip install -e ".[dev]"     # installs FAISS + Qdrant + serve extras too
ruff check . && pytest && python -m eval.retrieval_eval
```

CI installs the native backends, so the FAISS and Qdrant tests run there (they
skip only when those libraries are absent).

## License

MIT. See [LICENSE](LICENSE).
