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

**`rag_llm_infra.serve` has no authentication.** Installed with the `serve`
extra and used as the `Dockerfile`'s entrypoint, it exposes `/health`, `/index`
and `/query`. Any caller who can reach the port can replace the whole corpus via
`/index` and read back document text via `/query`; there is no key, no rate
limit and no bound on request size. It is a reference wiring of the library's
parts, not a hardened service — put it behind your own authentication, or do not
expose it.

**The JSON log formatter does not redact.** `log_config` forwards the fields a
caller attaches via `extra={...}` into the log line verbatim. It drops names
beginning with `_` and every field a bare `LogRecord` already carries; whatever
the caller added is emitted. The library reads no credentials of its own and logs
none, but it will faithfully log a secret you hand it — keep credentials out of
`extra`.
