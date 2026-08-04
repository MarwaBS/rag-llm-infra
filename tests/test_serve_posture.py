"""Pin the three boundaries SECURITY.md and the serve docstring describe: the
endpoints take no credential, the JSON formatter does not redact, and importing
the module configures neither logging nor tracing.

Each is asserted as it stands today rather than as it ought to be — an
undocumented change to any of them is what makes the prose wrong, so the change
has to break a test before it can ship.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rag_llm_infra.log_config as log_config
import rag_llm_infra.serve as serve

client = TestClient(serve.app)

SRC = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture(autouse=True)
def _reset_index():
    serve._index = None
    yield
    serve._index = None


def test_index_and_query_accept_requests_with_no_credential() -> None:
    assert client.post("/index", json={"documents": ["a document"]}).status_code == 201
    assert client.post("/query", json={"query": "document"}).status_code == 200


def _import_serve_and_report(prelude: str = "") -> str:
    """Import `serve` in a fresh interpreter; report what got configured.

    Observing the interpreter after a real import catches any route to
    configuration — an aliased import, an attribute call, a transitive one —
    where reading the module's own source only catches the shapes it parses for.
    """
    probe = (
        "import rag_llm_infra.serve, logging;"
        "import rag_llm_infra.log_config as lc, rag_llm_infra.tracing as tr;"
        "print(lc._CONFIGURED, tr._CONFIGURED, bool(logging.getLogger().handlers))"
    )
    result = subprocess.run(
        [sys.executable, "-c", prelude + probe],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    return result.stdout.strip()


def test_importing_serve_configures_neither_logging_nor_tracing() -> None:
    assert _import_serve_and_report() == "False False False"


def test_the_probe_reports_configuration_when_it_actually_happens() -> None:
    reported = _import_serve_and_report(
        "import rag_llm_infra.log_config as _lc; _lc.configure_logging();"
    )
    assert reported != "False False False", reported


def test_the_json_formatter_forwards_caller_supplied_fields_verbatim() -> None:
    record = logging.LogRecord("t", logging.INFO, "p", 1, "m", None, None)
    record.api_key = "sk-example"  # type: ignore[attr-defined]
    assert "sk-example" in log_config._JsonFormatter().format(record)
