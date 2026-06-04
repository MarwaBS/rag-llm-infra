# RAG + LLM Serving Infrastructure

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

An installable, vendor-neutral foundation for retrieval-augmented LLM applications:
a swappable vector store, a cached embedding index, a provider-agnostic LLM
protocol, the observability around them, a FastAPI serving layer, and a
retrieval-quality eval gate.

> Distilled infrastructure layer — typed, tested, packaged, and runnable on its own.

## Install

```bash
pip install rag-llm-infra                                   # core (numpy only)
pip install "rag-llm-infra[faiss,qdrant,openai,serve]"      # + native backends, OpenAI, serving
pip install -e ".[dev]"                                     # from source, for development
```

## Quickstart — end-to-end RAG (no API key, no network)

```bash
pip install rag-llm-infra
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
| `rag_llm_infra.evidence_index` | `EmbeddingEngine` — SentenceTransformers embeddings + adaptive, memory-pressure-aware LRU cache; reader/writer lock |
| `rag_llm_infra.tracing` | OpenTelemetry spans with console-exporter + no-op fallbacks |
| `rag_llm_infra.log_config` | structured JSON logging + an `llm_call` latency/token timer |
| `rag_llm_infra.serve` | FastAPI service (`/index`, `/query`, `/health`) wiring the parts together |

## Retrieval quality gate

```bash
python -m eval.retrieval_eval     # recall@1 / MRR on a labelled paraphrase corpus
```

Fails the build if retrieval quality regresses below threshold (`recall@1 ≥ 0.80`,
`MRR ≥ 0.85`). Wired into CI so a retrieval regression cannot merge.

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
