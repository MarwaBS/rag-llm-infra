"""Tests for tracing.py — OpenTelemetry distributed tracing.

The OTel API + SDK ship in the dev group, so the REAL configuration path
(provider setup, exporter selection, trace-context extraction) is exercised in
CI, not just the import-guarded fallbacks.
"""

from unittest.mock import MagicMock, patch

import pytest

try:
    import opentelemetry.sdk.trace  # noqa: F401

    OTEL_SDK_AVAILABLE = True
except ImportError:
    OTEL_SDK_AVAILABLE = False


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


@pytest.mark.skipif(not OTEL_SDK_AVAILABLE, reason="opentelemetry-sdk not installed")
class TestConfigureTracingWithSdk:
    """The real (non-fallback) configuration path."""

    def test_configure_tracing_sets_configured_flag(self):
        import rag_llm_infra.tracing as tracing

        original = tracing._CONFIGURED
        tracing._CONFIGURED = False
        try:
            tracing.configure_tracing(service_name="test-svc")
            assert tracing._CONFIGURED is True
        finally:
            tracing._CONFIGURED = original

    def test_otlp_endpoint_without_grpc_exporter_falls_back_to_console(self):
        """With an OTLP endpoint set but the grpc exporter package absent,
        configuration must degrade to the ConsoleSpanExporter and still complete
        (the degrade-don't-crash contract), not raise."""
        import rag_llm_infra.tracing as tracing

        original = tracing._CONFIGURED
        tracing._CONFIGURED = False
        try:
            with (
                patch.dict(
                    "os.environ",
                    {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"},
                ),
                patch.dict(
                    "sys.modules",
                    {"opentelemetry.exporter.otlp.proto.grpc.trace_exporter": None},
                ),
            ):
                tracing.configure_tracing(service_name="test-otlp-fallback")
            assert tracing._CONFIGURED is True
        finally:
            tracing._CONFIGURED = original


@pytest.mark.skipif(not OTEL_SDK_AVAILABLE, reason="opentelemetry-sdk not installed")
class TestCurrentTraceContextWithSdk:
    """current_trace_context against REAL spans — the contract log_config relies
    on to inject trace IDs into every log record."""

    def test_active_span_yields_valid_hex_ids(self):
        from opentelemetry.sdk.trace import TracerProvider

        from rag_llm_infra.tracing import current_trace_context

        # A local provider (not the global one) so this test is independent of
        # which test configured tracing first in this process.
        tracer = TracerProvider().get_tracer("test")
        with tracer.start_as_current_span("unit-span"):
            ctx = current_trace_context()
        assert len(ctx["trace_id"]) == 32
        assert len(ctx["span_id"]) == 16
        int(ctx["trace_id"], 16)  # valid lowercase hex, non-raising
        int(ctx["span_id"], 16)
        assert (
            ctx["trace_id"] != "0" * 32
        )  # a REAL recorded id, not the invalid zero id
        assert ctx["span_id"] != "0" * 16

    def test_no_active_span_yields_empty_strings(self):
        from rag_llm_infra.tracing import current_trace_context

        assert current_trace_context() == {"trace_id": "", "span_id": ""}


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
