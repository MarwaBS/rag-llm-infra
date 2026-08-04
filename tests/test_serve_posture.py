"""Pin the two boundaries SECURITY.md and the README describe.

Both are asserted as they stand today rather than as they ought to be: an
undocumented change to either is what makes the prose wrong, so the change has
to break a test before it can ship.
"""

import ast
import inspect
import logging

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


def _called_names(source: str) -> set[str]:
    """Names actually invoked — parsed, so a mention in prose cannot trip it."""
    return {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_serve_does_not_configure_logging_or_tracing_on_import() -> None:
    # Wiring either here would be inert anyway (see the next test), so the
    # module must not imply otherwise by calling them.
    called = _called_names(inspect.getsource(serve))
    assert not {"configure_logging", "configure_tracing"} & called


def test_the_call_detector_separates_a_call_from_a_mention() -> None:
    assert _called_names("configure_logging()") == {"configure_logging"}
    assert _called_names('"""call configure_logging() at startup."""') == set()


def test_configure_logging_is_inert_once_a_server_owns_the_root_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [logging.StreamHandler()])
    monkeypatch.setattr(log_config, "_CONFIGURED", False)
    monkeypatch.setattr(log_config, "ENV", "prod")
    log_config.configure_logging()
    installed = [type(h.formatter).__name__ for h in root.handlers]
    assert "_JsonFormatter" not in installed, installed


def test_the_json_formatter_forwards_caller_supplied_fields_verbatim() -> None:
    record = logging.LogRecord("t", logging.INFO, "p", 1, "m", None, None)
    record.api_key = "sk-example"  # type: ignore[attr-defined]
    assert "sk-example" in log_config._JsonFormatter().format(record)
