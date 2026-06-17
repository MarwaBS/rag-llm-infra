"""
OpenTelemetry distributed tracing configuration.

Usage (call once at application startup)::

    from tracing import configure_tracing
    configure_tracing()

Traces flow across the request path: retrieval → LLM call → downstream work.

Exporters (controlled by environment variables):
    OTEL_EXPORTER_OTLP_ENDPOINT  — set to send to Jaeger/Tempo/Honeycomb/etc.
                                    e.g. "http://localhost:4317"
    OTEL_SERVICE_NAME             — defaults to "rag-llm-service"

When OTEL_EXPORTER_OTLP_ENDPOINT is not set, a ConsoleSpanExporter is used
so traces are always visible in development without any external collector.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_CONFIGURED = False


def configure_tracing(service_name: str | None = None) -> None:
    """
    Set up the OpenTelemetry TracerProvider.
    Safe to call multiple times — only runs once.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:
        logger.warning(
            "opentelemetry-sdk not installed — tracing disabled. "
            "Run: pip install opentelemetry-sdk opentelemetry-api"
        )
        return

    svc = service_name or os.getenv("OTEL_SERVICE_NAME", "rag-llm-service")
    resource = Resource(attributes={SERVICE_NAME: svc})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            logger.info("OTel OTLP exporter configured endpoint=%s", otlp_endpoint)
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp-proto-grpc not installed — "
                "falling back to ConsoleSpanExporter."
            )
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()
        logger.info(
            "OTel ConsoleSpanExporter active (set OTEL_EXPORTER_OTLP_ENDPOINT for production)"
        )

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _CONFIGURED = True
    logger.info("OpenTelemetry tracing configured service=%s", svc)


def get_tracer(name: str = "rag-llm-service") -> Any:
    """Return a named tracer.  Returns a no-op tracer if OTel is unavailable."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


def current_trace_context() -> dict[str, str]:
    """
    Return {'trace_id': '...', 'span_id': '...'} for the active span.
    Returns empty strings when no span is active or OTel is unavailable.
    Used by log_config to inject trace IDs into every log record.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            return {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
            }
    except ImportError:
        pass
    return {"trace_id": "", "span_id": ""}


# ---------------------------------------------------------------------------
# No-op fallback so callers don't need to guard every `with tracer.start...`
# ---------------------------------------------------------------------------


class _NoOpSpan:
    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def set_attribute(self, *_: Any) -> None:
        pass

    def record_exception(self, *_: Any) -> None:
        pass

    def set_status(self, *_: Any) -> None:
        pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **_: Any) -> _NoOpSpan:
        return _NoOpSpan()
