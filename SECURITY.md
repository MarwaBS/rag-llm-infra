# Security Policy

## Supported versions

Only the latest published `0.1.x` release on PyPI receives fixes.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

This is a personal, open-source project; there is no formal security team.

If you find a security issue, please report it privately by emailing
**marwabensalem30@gmail.com** with the subject `[SECURITY] rag-llm-infra`.
Please do not open a public issue for a vulnerability before it is fixed.

I will acknowledge within 7 days and aim to ship a fix or documented mitigation
within 30 days, then publish a patched `0.1.x` release to PyPI.

## Scope

`rag-llm-infra` is a library: it runs in the caller's process with the caller's
inputs and provider credentials. It ships no server and stores no secrets.
Credentials (e.g. `OPENAI_API_KEY`) are read from the environment by the caller
and are never logged by the library. The optional Qdrant backend connects only
to the URL the caller supplies.
