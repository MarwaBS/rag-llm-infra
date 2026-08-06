"""Pin the three boundaries SECURITY.md and the serve docstring describe.

The endpoints take no credential. The JSON formatter does not redact. Neither
importing the module nor serving any route configures logging or tracing.
"""

import logging
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

import rag_llm_infra.log_config as log_config
import rag_llm_infra.serve as serve

client = TestClient(serve.app)


@pytest.fixture(autouse=True)
def _reset_index():
    serve._index = None
    yield
    serve._index = None


def test_index_and_query_accept_requests_with_no_credential() -> None:
    assert client.post("/index", json={"documents": ["a document"]}).status_code == 201
    assert client.post("/query", json={"query": "document"}).status_code == 200


# The package root is imported before the first snapshot: grpc and urllib3
# install handlers on the way in, and those are not serve's doing.
_SETUP = """
import logging
import logging.config
import rag_llm_infra
from opentelemetry import trace

def snapshot():
    root = logging.getLogger()
    touched = frozenset(
        (name, obj.level, len(obj.handlers))
        for name, obj in logging.Logger.manager.loggerDict.items()
        if isinstance(obj, logging.Logger) and (obj.handlers or obj.level)
    )
    return (root.level, len(root.handlers), touched,
            type(trace.get_tracer_provider()).__name__)

before = snapshot()
import rag_llm_infra.serve as serve
import rag_llm_infra.log_config as lc
import rag_llm_infra.tracing as tr
"""
_REPORT = "print(lc._CONFIGURED, tr._CONFIGURED, snapshot() == before)"
_UNTOUCHED = "False False True"

# Valid bodies first, then every declared route, so a new one needs no listing.
_EXERCISE = """
from fastapi.testclient import TestClient
with TestClient(serve.app) as c:
    c.post("/index", json={"documents": ["a document"]})
    c.post("/query", json={"query": "document"})
    for route in serve.app.routes:
        for method in sorted(getattr(route, "methods", None) or ()):
            c.request(method, route.path)
"""


def _report(body: str = "") -> str:
    """Run `serve` in a fresh interpreter; report what ended up configured."""
    result = subprocess.run(
        [sys.executable, "-c", _SETUP + body + _REPORT],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_importing_serve_configures_neither_logging_nor_tracing() -> None:
    assert _report() == _UNTOUCHED


def test_serving_every_route_configures_neither_logging_nor_tracing() -> None:
    assert _report(_EXERCISE) == _UNTOUCHED


@pytest.mark.parametrize(
    "body",
    [
        "lc.configure_logging()\n",
        "tr.configure_tracing()\n",
        "logging.basicConfig()\n",
        "logging.getLogger().setLevel(logging.DEBUG)\n",
        "logging.getLogger('somewhere').addHandler(logging.StreamHandler())\n",
        "logging.config.dictConfig({'version': 1, 'loggers': {'x': {'level': 'INFO'}}})\n",
        "trace.set_tracer_provider(__import__('opentelemetry.sdk.trace',"
        " fromlist=['TracerProvider']).TracerProvider())\n",
    ],
)
def test_the_probe_reports_configuration_when_it_actually_happens(body: str) -> None:
    assert _report(body) != _UNTOUCHED


def test_the_json_formatter_forwards_caller_supplied_fields_verbatim() -> None:
    record = logging.LogRecord("t", logging.INFO, "p", 1, "m", None, None)
    record.api_key = "sk-example"  # type: ignore[attr-defined]
    assert "sk-example" in log_config._JsonFormatter().format(record)
