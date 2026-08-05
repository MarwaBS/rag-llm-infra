"""Tests for log_config.py — structured logging and llm_call context manager."""

import contextlib
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
    """Line coverage of the timer is not assertion coverage: a constant is just
    as covered. The body cannot take less than it slept or more than the wall
    clock around it, and both of those are measured here rather than chosen —
    which leaves a constant and a scaled reading nowhere to sit. The 1 ms is
    clock granularity, not slack."""
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
    assert slept_ms - 1.0 <= reported <= wall_ms, records[0].llm


def _emit(**extra):
    """Format one record through the JSON formatter and return the payload."""
    from rag_llm_infra.log_config import _JsonFormatter

    record = logging.LogRecord("t", logging.INFO, "p", 1, "hello", None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return json.loads(_JsonFormatter().format(record))


def test_underscore_prefixed_fields_stay_out_of_the_log_line() -> None:
    """SECURITY.md tells callers that leading-underscore names are the ones the
    formatter drops. That sentence is only true while this holds."""
    payload = _emit(_internal="secret", visible="fine")
    assert "_internal" not in payload
    assert payload["visible"] == "fine"


def test_the_records_own_machinery_stays_out_of_the_log_line() -> None:
    """`message` and `asctime` are not attributes of a fresh record — a standard
    formatter stamps them on. A record that reached a second handler first
    carries them, and forwarding them would duplicate the message and the
    timestamp under a second name."""
    from rag_llm_infra.log_config import _JsonFormatter

    record = logging.LogRecord("t", logging.INFO, "p", 1, "hello", None, None)
    logging.Formatter("%(asctime)s %(message)s").format(record)
    assert hasattr(record, "message") and hasattr(record, "asctime")

    payload = json.loads(_JsonFormatter().format(record))
    assert "message" not in payload
    assert "asctime" not in payload
    assert payload["msg"] == "hello"


def test_a_failed_llm_call_is_logged_above_info(caplog) -> None:
    from rag_llm_infra.log_config import llm_call

    with caplog.at_level(logging.INFO, logger="llm"):
        with contextlib.suppress(RuntimeError):
            with llm_call("probe"):
                raise RuntimeError("provider down")
    record = next(r for r in caplog.records if r.message == "llm_call")
    assert record.levelno > logging.INFO
    assert record.llm["status"] == "error"
