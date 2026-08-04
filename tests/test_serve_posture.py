"""Pin the three boundaries SECURITY.md and the serve docstring describe: the
endpoints take no credential, the JSON formatter does not redact, and neither
importing the module nor serving a request configures logging or tracing.

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


_SETUP = """
import logging
import rag_llm_infra.serve as serve
import rag_llm_infra.log_config as lc
import rag_llm_infra.tracing as tr
"""
_REPORT = "print(lc._CONFIGURED, tr._CONFIGURED, bool(logging.getLogger().handlers))"

# Enter the client as a context manager so startup/lifespan handlers run, then
# serve both endpoints so anything configured lazily on first request has run too.
_EXERCISE = """
from fastapi.testclient import TestClient
with TestClient(serve.app) as c:
    c.post("/index", json={"documents": ["a document"]})
    c.post("/query", json={"query": "document"})
"""


def _report(body: str = "") -> str:
    """Run `serve` in a fresh interpreter; report what ended up configured.

    Observing the interpreter after the real thing has run catches any route to
    configuration — an aliased import, an attribute call, a getattr dispatch, a
    startup handler, a lazily imported submodule — where reading the module's
    own source only catches the shapes it is parsed for.
    """
    result = subprocess.run(
        [sys.executable, "-c", _SETUP + body + _REPORT],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    return result.stdout.strip()


def test_importing_serve_configures_neither_logging_nor_tracing() -> None:
    assert _report() == "False False False"


def test_serving_requests_configures_neither_logging_nor_tracing() -> None:
    assert _report(_EXERCISE) == "False False False"


@pytest.mark.parametrize(
    "body",
    [
        "lc.configure_logging()\n",
        "tr.configure_tracing()\n",
        "logging.getLogger().addHandler(logging.StreamHandler())\n",
    ],
)
def test_the_probe_reports_configuration_when_it_actually_happens(body: str) -> None:
    assert _report(body) != "False False False"


def test_the_json_formatter_forwards_caller_supplied_fields_verbatim() -> None:
    record = logging.LogRecord("t", logging.INFO, "p", 1, "m", None, None)
    record.api_key = "sk-example"  # type: ignore[attr-defined]
    assert "sk-example" in log_config._JsonFormatter().format(record)
