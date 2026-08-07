# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - unreleased

The suite and both eval gates have each been shown to go red, and so has the
replay that decides those verdicts.

`tests/mutations.json` registers the defects. `scripts/replay_mutations.py`
applies each one in turn. One test fails the build if the registry loses an
entry. Another fails it if a module in `eval/` has nothing registered against it.
That second check reads the `eval` package rather than the workflow files, so a
scoring gate that is not an eval module falls outside it.

The replay carries defects of its own and one control that must survive. A runner
stuck on green fails on the defects. One stuck on red fails on the control. The
control is pinned whole (file, anchor and replacement), so a live guard cannot
be parked in the registry and signed off as intended behaviour.

Credit is decided in one place. Only pytest's exit code for a failed test earns
it; 2, 3 and 4 are a collection error, an internal error and a bad invocation. A
mutation that stops the file loading is rejected rather than credited.

Lint, type-check, spell check and build run third-party tools over the tree and
carry no registered defect. `example.py` exits on an exception, not on a score.

### Security
- **`/index` and `/query` require a credential.** They read `X-API-Key` and
  compare it against `RAG_API_KEY` with `secrets.compare_digest`. With that
  variable unset they answer 503, so no configuration serves openly. Before
  this, anyone who reached the port could replace the corpus and read it back.
  `/health` stays open for container probes.
- **Two bounds on a request, because bytes on the wire do not bound memory.** A
  body over `RAG_MAX_BODY_BYTES` (1 MiB) is refused with 413 before it is read,
  and a POST with no `Content-Length` with 411. Measured before: a 20 MB body was
  accepted. Separately a corpus over `RAG_MAX_CORPUS_DOCS` (20000) is refused:
  every document becomes one `EMBED_DIM`-wide float32 row whatever its length, so
  262139 three-byte documents fit in a body under 1 MiB and materialise a 128 MiB
  matrix, 128x what the body bound admitted.
- **The credential path cannot be made to 500.** `secrets.compare_digest` raises
  `TypeError` on a non-ASCII `str`, and the header is attacker-controlled;
  comparison is on bytes now. A non-ASCII `RAG_API_KEY` is reported as
  misconfiguration, because headers arrive latin-1 while the environment is UTF-8
  and such a key could never match. A malformed `RAG_MAX_BODY_BYTES` falls back to
  the default with a warning rather than failing every request.
- **A route without the credential dependency fails the build.** The auth tests
  named the two routes that existed, so a third would have shipped open.
- **`pip-audit --strict` runs in CI and on the release path.** Its first run
  found four advisories: `starlette` 1.2.1 (PYSEC-2026-248, -249), reached
  through `fastapi`, and `h2` 4.3.0 (CVE-2026-71554), reached through
  `qdrant-client`'s `httpx[http2]`. Both now carry a floor in the extra that
  pulls them.
- **Every GitHub Action is pinned to a commit**, not a tag a publisher can
  repoint at other code.

### Added
- **`LLMProtocol` requires `close()` and `aclose()`.** Implementations hold
  network clients and the protocol had no way to release them, so a caller
  holding the protocol type could not close one. `FallbackLLM` closes every
  backend in its chain, including ones already tripped by budget exhaustion.
- **`.dockerignore`, denying by default.** `COPY . .` was admitting a 315 MiB
  virtualenv and the git history. The rules admit `pyproject.toml`, `README.md`,
  `LICENSE` and `src/` and nothing else; a test assembles a directory from those
  paths alone and builds the wheel in it, so a rule set too tight fails here
  rather than in the image.
- **CI builds the image, starts it, and exercises it.** The `Dockerfile`'s `CMD`
  and `HEALTHCHECK` were tracked text no workflow had run. The job now asserts an
  unauthenticated `/index` gets 401, that an authenticated index-then-query
  returns the expected document, and that the container's own health check
  reaches `healthy`.

### Removed: breaking
- **`CONFIG` and `RWLock` are no longer exported from the package root.** Every
  name in `__all__` is a compatibility promise and neither could be kept:
  `CONFIG` was read partly at construction and partly per call, and `RWLock` is
  an implementation detail of `EmbeddingEngine`. Both remain reachable at
  `rag_llm_infra.evidence_index`. A test pins the exported set.
- **`QdrantVectorStore` no longer defaults its collection to `"evidence"`.** The
  name is now the first, required argument, and `get_vector_store("qdrant")`
  needs `collection=`. `add()` deletes and recreates that collection, so two
  stores sharing a default name on one endpoint meant the second `add()`
  destroyed the first one's vectors. Measured before the change: store A wrote 3
  vectors, store B wrote 1 to the same default name, A reported `size == 3`
  against a collection holding 1, and `search(k=3)` answered `[0, -1, -1]`.
  Migration: pass the name your service owns.

### Fixed
- **A hanging provider held the whole fallback chain.** `FallbackLLM` advances
  when a backend raises, and a provider that blocks raises nothing. The failure
  the module exists to survive. Measured: `[slow(3s), fast]` returned the slow
  answer after 3.00s. Pass `timeout_s=` and a backend that does not answer in
  time raises `BackendTimeout`, which is retryable. Bounded: on the sync path
  this stops waiting, it does not cancel; the call continues on a daemon thread
  and its answer is discarded, because Python cannot interrupt a blocking socket
  read in another thread. The async path uses `asyncio.wait_for`, which cancels.
- **Memory pressure raised the cache limit.** `max(100, limit * 0.5)` meant any
  configured limit below 200 grew under pressure (measured: 50 became 100), and
  it never came back. The trim now halves the configured ceiling, never exceeds
  it, and restores it when pressure clears.
- **Equal scores came back in whatever order the partition left them.**
  `np.argsort` defaults to quicksort, which is not stable. NumPy now orders ties
  by the lower document index. The alternative, a full `lexsort`, which would
  also fix *which* tied documents are selected, was measured first. It would run
  on every query, not only tied ones, and on distinct scores at a million
  documents it costs about 20x (21.7x on the machine that measured it; 1.8x when
  every score ties). So the selection stays unspecified and the protocol says so
  rather than implying otherwise. Reproduce with `benchmarks/topk_tie_cost.py`,
  which prints its interpreter, numpy version and platform.
- **The JSON log line carried logging's own fields.** The formatter dropped a
  hand-written list of `LogRecord` attributes, a blacklist over an open set, so
  Python 3.12's `taskName` appeared on every line as `null`. The exclusion set is
  now read from a bare `LogRecord`, so the next such field is excluded on
  arrival. `ts` is UTC with an offset and milliseconds; it was local time to the
  second, which cannot be ordered across hosts.
- **The broad exception handlers say what they mean.** The memory-pressure trim
  logs when it skips instead of passing silently. Each optional-dependency probe
  (`faiss`, `psutil`, `qdrant-client`, `sentence-transformers`) reports a library
  that is installed but will not load, at warning level, and still degrades to
  the fallback. Measured on Windows: a bad extension module raises `ImportError`
  from the loader, but an explicit `ctypes` load inside a package (the shape
  `torch` and `sentence-transformers` use) raises `OSError`, which
  `except ImportError` alone would let escape and stop `import rag_llm_infra`.
  The trace-context lookup stays broad and cannot report anything: it runs inside
  `format()`, where raising loses the record and logging recurses.
- **`QdrantVectorStore.size` counted nothing.** It returned a number remembered
  from the last `add()`. `search` derives its row width from it, so once the
  count was stale the answer was padded with `-1` sentinel indices. It now counts
  the collection, at the cost of one round-trip.
- **`OpenAIBackend.close()` closed nothing on the async client.** `AsyncOpenAI.close`
  is a coroutine function, so calling it from a sync method discarded the
  coroutine and left the httpx pool open. `close()` now handles the sync client
  only and **`aclose()` is added** for the async one. Both propagate a failure to
  close instead of swallowing it, so `close()` can now raise where it could not.
- **The generation gate could not fail.** Its faithful fixture was a retrieved
  document verbatim, scoring `1.000` by set identity whatever the metric did.
  Both sides are now populations of paraphrases and of fluent unsupported
  answers, scored by the worst case in each, and a test forbids any fixture
  lifted from its own evidence.
- **No eval floor is edited where it is used.** `scripts/derive_eval_floors.py`
  measures the labelled populations and writes `eval/eval_floors.json`, which
  both gates read; re-running the producer must reproduce it byte for byte. It
  refuses to derive at all unless the metric separates the two populations.
- **`groundedness` is pinned to exact fractions** taken from its written
  definition rather than to `0 < score < 1`, which a metric that has stopped
  reading the evidence also satisfies.
- **The replay credited reds it had not earned.** A mutation that broke the
  import reddened every gate that touched the file without removing any
  behaviour, and a gate that could not run at all counted the same as a test that
  failed. The mutated file must now still import, and only pytest's exit code for
  a failed test earns credit; 2, 3 and 4 do not.
- **The demo embedder had no test.** Both eval gates, `example.py` and the
  serving demo embed through it, and case folding, term frequency, digit tokens
  and cross-process reproducibility could each be removed with every gate green.

### Changed
- **`auto` prefers FAISS for a measured reason.** `benchmarks/backend_crossover.py`
  times both backends at 384 dimensions. FAISS is ahead at every size the script
  tries, from a hundred documents to a hundred thousand. Bounded to the host the
  script prints, which was CPython 3.13.13 with numpy 2.4.6 on Windows 11.
- **The message shape is a `TypedDict`, and the type gate covers `eval` and
  `tests`.** `list[dict[str, Any]]` accepted a misspelled key, an integer role
  and a message with no role; `ignore_missing_imports` was global, which also
  silenced a typo in a first-party import path. Both are fixed, and
  `mypy src eval tests` runs in CI. It reported 49 errors the first time it was
  pointed at those trees.
- `requires-python` is bounded at both ends (`>=3.12,<3.14`) and CI runs both
  legs it admits.
- **Documentation states mechanisms and is executed where it can be.** The
  README's request bodies are posted through the app by a test and its relative
  links are resolved; `codespell` runs as a CI gate; `mypy` covers `scripts/`;
  and `llm_call` no longer claims to measure tokens, which it never did.
- Documentation states what the code does: what the log formatter does not
  redact, and that importing
  `rag_llm_infra.serve` configures neither logging nor tracing (pinned by a
  subprocess probe with seven positive controls).

## [0.1.2] - 2026-07-04

Hardening release. Each behavioral fix carries a regression test.

### Fixed
- **Packaging: the `py.typed` marker now ships in the wheel.** Without it (PEP
  561) downstream mypy/pyright silently ignored every type hint the package
  exports, despite the README selling a typed API.
- **Empty-store `search()` now behaves uniformly across backends.** Building a
  store from a zero-row `add()` and then searching used to diverge three ways:
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
  query/index dimension mismatch, uniformly across FAISS/NumPy/Qdrant.
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
  configuration path, the OTLP-endpoint->console-exporter degradation, and
  valid trace/span IDs from `current_trace_context` inside a live span.
- **`CHANGELOG.md`** (this file).

### Changed
- **`groundedness` documents its blind spots instead of overselling.** The module
  docstring and README now state plainly that the lexical metric is
  negation-blind, dilutable, and scores vocabulary rather than propositions: a
  cheap tripwire, not a faithfulness guarantee. Tests pin those limits so a
  later edit can't quietly claim more.
- **`serve.py` sources the FastAPI `version` from `__version__`** instead of a
  hardcoded `"0.1.0"` that had already drifted from the released package.
- **`FallbackLLM` documents its thread-safety contract** and advances its
  budget-exhaustion high-water mark monotonically (`max(...)`).
- **Release workflow is gated and uses build-once / promote.** Publishing on a
  `v*` tag now runs the full ruff / format / mypy / pytest / eval suite first,
  asserts the tag matches the package version, then builds the wheel + sdist and
  validates *that artifact* (clean-venv install, import, py.typed-ships check).
  The publish job downloads and uploads those exact bytes via a workflow
  artifact instead of rebuilding, so the wheel that reaches PyPI is the one the
  gate validated, not a fresh, untested build. Upload stays `--skip-existing`
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
- Two-sided faithfulness eval gate: an absolute ceiling on the hallucinated
  control, and the margin requirement raised from merely positive to `0.50`,
  alongside the existing floor on the faithful answer. Both fixtures were single
  answers and the faithful one was a retrieved document verbatim. See 0.2.0 for
  what replaced them.
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

[0.2.0]: https://github.com/MarwaBS/rag-llm-infra/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/MarwaBS/rag-llm-infra/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MarwaBS/rag-llm-infra/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MarwaBS/rag-llm-infra/releases/tag/v0.1.0
