# RAG + LLM Serving Infrastructure

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rag-llm-infra)](https://pypi.org/project/rag-llm-infra/)

An installable, vendor-neutral foundation for retrieval-augmented LLM applications:
a swappable vector store, a cached embedding index, a provider-agnostic LLM
protocol, the observability around them, a FastAPI serving layer, and a
retrieval-quality eval gate.

> Distilled infrastructure layer — typed, tested, packaged, and runnable on its own.

## Install

```bash
pip install rag-llm-infra                                   # core (numpy)
pip install "rag-llm-infra[faiss,qdrant,openai,serve]"      # + native backends, OpenAI, serving
pip install "rag-llm-infra[psutil]"                         # + memory-pressure-aware cache trimming
pip install -e ".[dev]"                                     # from a local clone, for development
```

## Quickstart — end-to-end RAG (no API key, no network)

```bash
git clone https://github.com/MarwaBS/rag-llm-infra && cd rag-llm-infra
pip install -e .
python example.py
```

```
embed documents → index in a VectorStore → retrieve top-k for a query
                → build a grounded prompt → answer with an LLMProtocol backend
```

Runs on the NumPy vector store + the deterministic mock LLM, so it needs no key.
In production, swap the demo embedder for `EmbeddingEngine` and `get_llm("mock")`
for `get_llm("openai")`.

## Serve it

```bash
pip install "rag-llm-infra[serve]"
uvicorn rag_llm_infra.serve:app          # or: docker build -t rag-llm-infra . && docker run -p 8000:8000 rag-llm-infra
```

```bash
curl -XPOST localhost:8000/index -d '{"documents":["FAISS is in-process vector search","Qdrant is a vector database"]}' -H 'content-type: application/json'
curl -XPOST localhost:8000/query -d '{"query":"vector search","k":1}'      -H 'content-type: application/json'
```

## What's inside

| Module | Responsibility |
| --- | --- |
| `rag_llm_infra.llm_protocol` | `LLMProtocol` — `runtime_checkable` Protocol over OpenAI / Anthropic-stub / Mock; factory `get_llm()` |
| `rag_llm_infra.vector_store` | `VectorStoreProtocol` — in-process FAISS `IndexFlatIP`, pure-NumPy fallback, real **Qdrant** (batched search) |
| `rag_llm_infra.evidence_index` | `EmbeddingEngine` — SentenceTransformers embeddings + a cache (insertion-order eviction) guarded by a writer-preferring reader/writer lock, so the slow `model.encode` runs outside the lock. Memory-pressure-aware trimming activates with the `[psutil]` extra (`pip install "rag-llm-infra[psutil]"`); without it the cache is fixed-size |
| `rag_llm_infra.tracing` | OpenTelemetry spans with console-exporter + no-op fallbacks |
| `rag_llm_infra.log_config` | structured JSON logging + an `llm_call` latency/token timer |
| `rag_llm_infra.serve` | FastAPI service (`/index`, `/query`, `/health`) wiring the parts together |
| `rag_llm_infra.faithfulness` | `groundedness(answer, contexts)` — lexical faithfulness metric for RAG output |
| `rag_llm_infra.fallback` | `FallbackLLM` — budget-aware multi-provider routing; drop-in `LLMProtocol` |

## Quality gates

```bash
python -m eval.retrieval_eval      # recall@1 / MRR — retrieval mechanics over the demo embedder
python -m eval.generation_eval     # groundedness (faithfulness) of generated answers
```

Both run in CI: a **retrieval** regression (`recall@1 ≥ 0.80`, `MRR ≥ 0.85`) or a
**faithfulness** regression (grounded answer below threshold, or the metric failing
to flag a hallucinated control) fails the build and cannot merge.

`groundedness` is a **cheap lexical tripwire, not a faithfulness guarantee** — it
scores token overlap, so by construction it is negation-blind ("X is not Y" looks
grounded), dilutable (a false clause appended to a true answer only dents the
score), and propositional claims it can't verify. It catches the common
out-of-vocabulary hallucination signature cheaply on every generation; pair it with
an LLM-judge for semantic faithfulness. The limits are spelled out in the
`faithfulness` module docstring and pinned by tests so they can't be quietly
oversold later.

## Engineering principles demonstrated

- **Swap by interface** — `LLMProtocol` / `VectorStoreProtocol` make the model and the index runtime-swappable.
- **Degrade, don't crash** — FAISS / Qdrant / OpenTelemetry / SentenceTransformers are lazily imported with working fallbacks; missing infra never hard-fails import.
- **Measured, not asserted** — a retrieval eval gate, not just unit tests; packaged and CI-built end to end.

## Develop / test

```bash
pip install -e ".[dev]"     # installs FAISS + Qdrant + serve extras too
ruff check . && pytest && python -m eval.retrieval_eval
```

CI installs the native backends, so the FAISS and Qdrant tests run there (they
skip only when those libraries are absent).

## License

MIT — see [LICENSE](LICENSE).
