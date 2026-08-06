"""The JSON line carries the caller's fields and none of logging's own.

The exclusion set is read from a bare `LogRecord` rather than written down. A
hand-written list is a blacklist over an open set: Python 3.12 added `taskName`
to every record and each line began carrying `"taskName": null`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pytest

from rag_llm_infra.log_config import _RECORD_OWN_FIELDS, _JsonFormatter


def _emit(**extras: object) -> dict:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello", None, None)
    for key, value in extras.items():
        setattr(record, key, value)
    return json.loads(_JsonFormatter().format(record))


def test_the_exclusion_set_is_read_from_a_real_record() -> None:
    bare = logging.LogRecord("", 0, "", 0, "", None, None)
    assert set(bare.__dict__) <= _RECORD_OWN_FIELDS
    assert {"message", "asctime"} <= _RECORD_OWN_FIELDS


def test_no_logging_internal_reaches_the_output() -> None:
    emitted = set(_emit())
    assert emitted == {
        "ts",
        "level",
        "logger",
        "msg",
        "trace_id",
        "span_id",
        "request_id",
    }


@pytest.mark.parametrize(
    "field",
    ["taskName", "pathname", "lineno", "process", "threadName", "relativeCreated"],
)
def test_a_named_record_internal_is_absent(field: str) -> None:
    assert field not in _emit()


def test_the_control_shows_caller_fields_are_forwarded() -> None:
    emitted = _emit(request_id="abc", llm={"tokens": 5})
    assert emitted["request_id"] == "abc"
    assert emitted["llm"] == {"tokens": 5}


def test_underscore_prefixed_fields_are_dropped() -> None:
    assert "_secret" not in _emit(_secret="value")


def test_the_timestamp_is_timezone_aware_and_sub_second() -> None:
    parsed = datetime.fromisoformat(_emit()["ts"])
    assert parsed.tzinfo is not None
    assert parsed.microsecond or "." in _emit()["ts"]
