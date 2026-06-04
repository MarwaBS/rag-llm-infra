"""
Structured logging configuration.

Usage (call once at application startup)::

    from log_config import configure_logging
    configure_logging()

In production (ENV=prod) log records are emitted as single-line JSON objects
so they can be ingested by log-aggregation systems (CloudWatch, Datadog, etc.).
In development the default human-readable format is used.

Also provides an ``llm_call`` context manager that measures latency and token
usage for every LLM invocation and emits a structured summary::

    with llm_call("expand_summary", model="gpt-4o") as ctx:
        result = chain.invoke(inputs)
        ctx["tokens"] = result.usage_metadata.get("total_tokens", 0)
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

# Injected at format-time so every log record carries the active trace/span IDs.
def _get_trace_context() -> dict[str, str]:
    try:
        from tracing import current_trace_context
        return current_trace_context()
    except Exception:
        return {"trace_id": "", "span_id": ""}

ENV: str = os.getenv("ENV", "dev").lower()
_CONFIGURED = False


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit each record as a single-line JSON object."""

    _SKIP = frozenset(logging.LogRecord.__init__.__code__.co_varnames)

    def format(self, record: logging.LogRecord) -> str:
        trace_ctx = _get_trace_context()
        # request_id is attached by the caller via `extra={"request_id": ...}`
        _request_id = getattr(record, "request_id", "")
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_ctx["trace_id"],
            "span_id": trace_ctx["span_id"],
            "request_id": _request_id,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Forward any extra={} fields attached by the caller
        for key, val in record.__dict__.items():
            if key.startswith("_") or key in {
                "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "name",
                "message", "asctime",
            }:
                continue
            payload[key] = val
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    """Configure root logger.  Safe to call multiple times — only runs once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    if root.handlers:
        # Already configured externally (e.g. pytest's log capturing)
        _CONFIGURED = True
        return
    handler = logging.StreamHandler()
    if ENV == "prod":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _CONFIGURED = True


@contextmanager
def llm_call(
    operation: str,
    model: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Generator[dict[str, Any], None, None]:
    """Measure latency + tokens for a single LLM call and log the result.

    Example::

        with llm_call("expand_summary", model="gpt-4o") as ctx:
            result = chain.invoke(inputs)
            ctx["tokens"] = result.usage_metadata.get("total_tokens", 0)

    The context dict ``ctx`` is yielded before the body runs; callers can
    attach any extra fields.  On exit the manager logs a structured summary
    whether or not an exception occurred.
    """
    _log = logger or logging.getLogger("llm")
    ctx: dict[str, Any] = {
        "operation": operation,
        "model": model or os.getenv("OPENAI_MODEL_MAIN", "gpt-4o"),
        "tokens": 0,
    }
    start = time.perf_counter()
    try:
        yield ctx
        ctx.setdefault("status", "ok")
    except Exception as exc:
        ctx["status"] = "error"
        ctx["error"] = str(exc)
        raise
    finally:
        ctx["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        level = logging.WARNING if ctx.get("status") == "error" else logging.INFO
        _log.log(level, "llm_call", extra={"llm": ctx})
