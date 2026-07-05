# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-07-04

Audit-driven hardening. Each behavioral fix carries a regression test.

### Fixed
- **Packaging: the `py.typed` marker now ships in the wheel.** Without it (PEP
  561) downstream mypy/pyright silently ignored every type hint the package
  exports, despite the README selling a typed API.
- **Empty-store `search()` now behaves uniformly across backends.** Building a
  store from a zero-row `add()` and then searching used to diverge three ways —
  FAISS raised a bare `AssertionError`, Qdrant a misleading
  `"called before add()"`, and only NumPy honoured the documented `min(k, size)`
  contract. All three now return `(Nq, 0)`-shaped arrays. Calling `search`
  *before any* `add` remains a `RuntimeError` (a programming error) on every
  backend. Parametrized regression test across FAISS/NumPy/Qdrant.
- **`groundedness` no longer drops two-character tokens.** The `len > 2` filter
  made acronyms (US, AI, ML, UK) invisible, so an answer built from them scored a
  vacuous `1.0` regardless of the evidence. Now `len >= 2`.

### Added
- **Input validation on every vector-store backend.** `add`/`search` raise a
  clear `ValueError` for a non-2-D array (was an opaque `AxisError`), for
  non-finite (NaN/inf) embeddings (was silent garbage scores), and for a
  query/index dimension mismatch — uniformly across FAISS/NumPy/Qdrant.
- **Within-batch deduplication in `EmbeddingEngine.embed_batch`.** A text
  repeated inside one batch is now encoded once, not once per occurrence
  (deduplication previously happened only against the cache, not within the
  batch). Regression test asserts one encode per unique text.
- **`[psutil]` extra activating the memory-pressure-aware cache trim.** psutil
  was previously undeclared in every dependency group, so the advertised
  trimming was an unreachable branch in any documented install. It is now an
  optional extra (`pip install "rag-llm-infra[psutil]"`), ships in the dev
  group so CI runs the real branch, and the trim/no-trim behavior is pinned by
  regression tests (oldest entries actually evicted under pressure; nothing
  evicted without it).
- **Coverage gate in CI and the release gate** (`--cov-fail-under=85`; measured
  93% when introduced), plus real-SDK tracing tests: the OpenTelemetry
  configuration path, the OTLP-endpoint→console-exporter degradation, and
  valid trace/span IDs from `current_trace_context` inside a live span.
- **`CHANGELOG.md`** (this file).

### Changed
- **`groundedness` documents its blind spots instead of overselling.** The module
  docstring and README now state plainly that the lexical metric is
  negation-blind, dilutable, and scores vocabulary rather than propositions — a
  cheap tripwire, not a faithfulness guarantee — and tests pin those limits so a
  later edit can't quietly claim more.
- **`serve.py` sources the FastAPI `version` from `__version__`** instead of a
  hardcoded `"0.1.0"` that had already drifted from the released package.
- **`FallbackLLM` documents its thread-safety contract** and advances its
  budget-exhaustion high-water mark monotonically (`max(...)`), so a concurrent
  call can never regress it.
- **Release workflow is gated and uses build-once / promote.** Publishing on a
  `v*` tag now runs the full ruff / format / mypy / pytest / eval suite first,
  asserts the tag matches the package version, then builds the wheel + sdist and
  validates *that artifact* (clean-venv install, import, py.typed-ships check).
  The publish job downloads and uploads those exact bytes via a workflow
  artifact instead of rebuilding, so the wheel that reaches PyPI is the one the
  gate validated — not a fresh, untested build. Upload stays `--skip-existing`
  (idempotent re-push) and now runs `twine check` first.
- **Ruff lint tightened** to `F, E, I, B, UP` (import sorting, bugbear, and
  pyupgrade on top of pyflakes/pycodestyle); the codebase was modernized to the
  py3.12 syntax floor (PEP 604 unions, PEP 585 generics, PEP 695 type aliases).

## [0.1.1] - 2026-05

### Added
- CI workflow running ruff, `ruff format --check`, mypy, pytest, the retrieval
  and generation eval gates, the wheel/sdist build, and the end-to-end example.
- Real, tested Qdrant backend (`QdrantVectorStore`) replacing the Pinecone stub,
  proving the `VectorStoreProtocol` swap path end-to-end with batched search.
- Two-sided faithfulness eval gate: the faithful answer must score above a high
  threshold AND the hallucinated control below a low one, so the gate can fail
  (the earlier construction made groundedness identically `1.0`).
- Budget-aware `FallbackLLM` with a permanent budget-exhaustion trip.

### Fixed
- Vector-store `search` row width is `min(k, size)` across backends (no FAISS
  `-1`/`-inf` padding); `add` no longer mutates the caller's array in place.

## [0.1.0] - 2026-05

### Added
- Initial public release on PyPI.
- `LLMProtocol` + factory (`OpenAIBackend`, `AnthropicBackend` stub,
  `MockBackend`), `VectorStoreProtocol` (FAISS/NumPy), cached `EmbeddingEngine`
  with a writer-preferring reader/writer lock, OpenTelemetry tracing helpers,
  structured logging, and a FastAPI service.
- MIT license.

[0.1.2]: https://github.com/MarwaBS/rag-llm-infra/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MarwaBS/rag-llm-infra/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MarwaBS/rag-llm-infra/releases/tag/v0.1.0
