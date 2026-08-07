# Security Policy

## Supported versions

Only the latest published minor release on PyPI receives fixes.

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅        |
| < 0.2   | ❌        |

## Reporting a vulnerability

This is a personal, open-source project; there is no formal security team.

If you find a security issue, please report it privately by emailing
**marwabensalem30@gmail.com** with the subject `[SECURITY] rag-llm-infra`.
Please do not open a public issue for a vulnerability before it is fixed.

I will acknowledge within 7 days and aim to ship a fix or documented mitigation
within 30 days, then publish a patched release to PyPI.

## Scope

`rag-llm-infra` is primarily a library: it runs in the caller's process with the
caller's inputs and provider credentials, and stores no secrets. The optional
Qdrant backend connects only to the URL the caller supplies.

Two caveats qualify that.

**`rag_llm_infra.serve` authenticates with a shared key, and nothing else.**
Installed with the `serve` extra and used as the `Dockerfile`'s entrypoint, it
serves `/index` and `/query` behind a credential and `/health`, `/docs`, `/redoc`
and `/openapi.json` without one. The two POST routes require `X-API-Key` to equal
`RAG_API_KEY`, compared as bytes with `secrets.compare_digest`, and answer 503
when that variable is unset — there is no configuration in which they serve
openly. A test pins that open list exactly and requires every other route to hold
the credential dependency by identity. `/health` returns only `{"status": "ok"}`;
the other three describe the API's shape, not its data.

Two bounds. A request body over `RAG_MAX_BODY_BYTES` (1 MiB by default) is
refused with 413 before being read, and a POST with no `Content-Length` with 411.
A corpus over `RAG_MAX_CORPUS_DOCS` (20000) is refused with 413 as well: each
document becomes a fixed-width float32 row, so a body inside the byte bound can
still materialise a matrix two orders of magnitude larger.

What it still does not have: rate limiting, per-caller identity, key rotation,
audit logging, or TLS. One key holder is every key holder, and `/index` replaces
the whole corpus. It is a reference wiring of the library's parts, not a
multi-tenant service.

**The JSON log formatter does not redact.** `log_config` forwards the fields a
caller attaches via `extra={...}` into the log line verbatim. It drops names
beginning with `_` and every field a bare `LogRecord` already carries; whatever
the caller added is emitted. The library reads no credentials of its own and logs
none, but it will faithfully log a secret you hand it — keep credentials out of
`extra`.
