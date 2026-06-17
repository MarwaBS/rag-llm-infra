"""Tests for tracing.py — OpenTelemetry distributed tracing."""

from unittest.mock import MagicMock, patch

import pytest


class TestConfigureTracing:
    def test_configure_tracing_idempotent(self):
        import rag_llm_infra.tracing as tracing

        original = tracing._CONFIGURED
        tracing._CONFIGURED = False
        try:
            tracing.configure_tracing(service_name="test-svc")
            # On envs without opentelemetry-sdk, _CONFIGURED stays False (graceful)
            first_state = tracing._CONFIGURED
            # Second call should not change state
            tracing.configure_tracing()
            assert tracing._CONFIGURED == first_state
        finally:
            tracing._CONFIGURED = original

    def test_configure_tracing_without_otel(self):
        import rag_llm_infra.tracing as tracing

        original = tracing._CONFIGURED
        tracing._CONFIGURED = False
        try:
            with patch.dict(
                "sys.modules",
                {
                    "opentelemetry": None,
                    "opentelemetry.trace": None,
                    "opentelemetry.sdk": None,
                    "opentelemetry.sdk.trace": None,
                    "opentelemetry.sdk.trace.export": None,
                    "opentelemetry.sdk.resources": None,
                },
            ):
                tracing.configure_tracing()
                # Should NOT set _CONFIGURED because OTel import failed
                assert tracing._CONFIGURED is False
        finally:
            tracing._CONFIGURED = original

    def test_configure_tracing_with_otlp_endpoint(self):
        import rag_llm_infra.tracing as tracing

        original = tracing._CONFIGURED
        tracing._CONFIGURED = False
        try:
            with patch.dict(
                "os.environ", {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"}
            ):
                tracing.configure_tracing(service_name="test")
                # On envs without opentelemetry-sdk, _CONFIGURED stays False
        finally:
            tracing._CONFIGURED = original


class TestGetTracer:
    def test_returns_tracer(self):
        from rag_llm_infra.tracing import get_tracer

        tracer = get_tracer("test")
        assert tracer is not None

    def test_returns_noop_when_otel_missing(self):
        from rag_llm_infra.tracing import _NoOpTracer, get_tracer

        with patch.dict(
            "sys.modules",
            {"opentelemetry": None, "opentelemetry.trace": None},
        ):
            tracer = get_tracer()
            assert isinstance(tracer, _NoOpTracer)


class TestCurrentTraceContext:
    def test_returns_dict_with_keys(self):
        from rag_llm_infra.tracing import current_trace_context

        ctx = current_trace_context()
        assert "trace_id" in ctx
        assert "span_id" in ctx

    def test_returns_empty_strings_without_otel(self):
        from rag_llm_infra.tracing import current_trace_context

        with patch.dict(
            "sys.modules",
            {"opentelemetry": None, "opentelemetry.trace": None},
        ):
            ctx = current_trace_context()
        assert ctx == {"trace_id": "", "span_id": ""}


class TestNoOpSpan:
    def test_context_manager(self):
        from rag_llm_infra.tracing import _NoOpSpan

        span = _NoOpSpan()
        with span as s:
            s.set_attribute("key", "value")
            s.record_exception(Exception("test"))
            s.set_status("OK")

    def test_enter_returns_self(self):
        from rag_llm_infra.tracing import _NoOpSpan

        span = _NoOpSpan()
        assert span.__enter__() is span


class TestNoOpTracer:
    def test_start_as_current_span(self):
        from rag_llm_infra.tracing import _NoOpSpan, _NoOpTracer

        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test")
        assert isinstance(span, _NoOpSpan)

    def test_span_as_context_manager(self):
        from rag_llm_infra.tracing import _NoOpTracer

        tracer = _NoOpTracer()
        with tracer.start_as_current_span("test") as span:
            span.set_attribute("x", 1)
