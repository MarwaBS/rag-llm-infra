"""Tests for log_config.py — structured logging and llm_call context manager."""

import json
import logging
from unittest.mock import patch

import pytest


class TestJsonFormatter:
    def test_format_produces_valid_json(self):
        from rag_llm_infra.log_config import _JsonFormatter

        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert data["msg"] == "hello world"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert "ts" in data

    def test_format_includes_exception(self):
        from rag_llm_infra.log_config import _JsonFormatter

        fmt = _JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="fail",
            args=(),
            exc_info=exc_info,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert "exc" in data
        assert "ValueError" in data["exc"]

    def test_format_includes_extra_fields(self):
        from rag_llm_infra.log_config import _JsonFormatter

        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"
        output = fmt.format(record)
        data = json.loads(output)
        assert data["custom_field"] == "custom_value"

    def test_format_includes_trace_context(self):
        from rag_llm_infra.log_config import _JsonFormatter

        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert "trace_id" in data
        assert "span_id" in data


class TestConfigureLogging:
    def test_configure_logging_runs_once(self):
        import rag_llm_infra.log_config as log_config

        original = log_config._CONFIGURED
        log_config._CONFIGURED = False
        try:
            # With existing handlers (pytest adds them), it should just mark configured
            log_config.configure_logging()
            assert log_config._CONFIGURED is True
            # Second call is no-op
            log_config.configure_logging()
            assert log_config._CONFIGURED is True
        finally:
            log_config._CONFIGURED = original

    def test_configure_logging_prod_mode(self):
        import rag_llm_infra.log_config as log_config

        original_configured = log_config._CONFIGURED
        original_env = log_config.ENV
        log_config._CONFIGURED = False
        log_config.ENV = "prod"
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers = []  # Clear handlers to trigger setup
        try:
            log_config.configure_logging()
            # Should have added a handler with JsonFormatter
            added = [h for h in root.handlers if h not in original_handlers]
            if added:
                assert isinstance(added[0].formatter, log_config._JsonFormatter)
        finally:
            root.handlers = original_handlers
            log_config._CONFIGURED = original_configured
            log_config.ENV = original_env

    def test_configure_logging_dev_mode(self):
        import rag_llm_infra.log_config as log_config

        original_configured = log_config._CONFIGURED
        original_env = log_config.ENV
        log_config._CONFIGURED = False
        log_config.ENV = "dev"
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers = []
        try:
            log_config.configure_logging()
            added = [h for h in root.handlers if h not in original_handlers]
            if added:
                assert not isinstance(added[0].formatter, log_config._JsonFormatter)
        finally:
            root.handlers = original_handlers
            log_config._CONFIGURED = original_configured
            log_config.ENV = original_env


class TestLlmCall:
    def test_successful_call(self):
        from rag_llm_infra.log_config import llm_call

        with llm_call("test_op", model="gpt-4o") as ctx:
            ctx["tokens"] = 100
        assert ctx["status"] == "ok"
        assert ctx["tokens"] == 100
        assert "latency_ms" in ctx
        assert ctx["latency_ms"] >= 0

    def test_failed_call(self):
        from rag_llm_infra.log_config import llm_call

        with pytest.raises(ValueError):
            with llm_call("test_op") as ctx:
                raise ValueError("boom")
        assert ctx["status"] == "error"
        assert ctx["error"] == "boom"
        assert "latency_ms" in ctx

    def test_default_model(self):
        from rag_llm_infra.log_config import llm_call

        with llm_call("test_op") as ctx:
            pass
        assert ctx["model"] is not None

    def test_custom_logger(self):
        from rag_llm_infra.log_config import llm_call

        custom_logger = logging.getLogger("custom_test")
        with llm_call("test_op", logger=custom_logger) as ctx:
            pass
        assert ctx["status"] == "ok"


def test_llm_call_reports_a_measured_latency(caplog) -> None:
    """Line coverage of the timer is not assertion coverage: a constant would be
    just as covered. Bound the reported number on both sides against a clock this
    test owns — a lower bound alone accepts any number large enough."""
    import time as _time

    from rag_llm_infra.log_config import llm_call

    slept_ms = 50.0
    started = _time.perf_counter()
    with caplog.at_level(logging.INFO, logger="llm"):
        with llm_call("probe"):
            _time.sleep(slept_ms / 1000)
    wall_ms = (_time.perf_counter() - started) * 1000

    records = [r for r in caplog.records if r.message == "llm_call"]
    assert len(records) == 1
    reported = records[0].llm["latency_ms"]
    assert 0.8 * slept_ms <= reported <= wall_ms, records[0].llm
