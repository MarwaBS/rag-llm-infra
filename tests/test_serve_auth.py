"""The corpus is not reachable without a credential, and a body cannot be unbounded.

`/index` replaces the whole corpus and `/query` reads it back, so an open port
was a full read-write handle on the service's data. The bound matters because the
corpus lives in this process: whatever a request carries, the process holds.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import rag_llm_infra.serve as serve
from rag_llm_infra.serve import DEFAULT_MAX_BODY_BYTES, app

KEY = "correct-horse"
client = TestClient(app)
authed = TestClient(app, headers={"X-API-Key": KEY})


@pytest.fixture(autouse=True)
def _empty_index():
    serve._index = None
    yield
    serve._index = None


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", KEY)


@pytest.mark.parametrize(
    "path,body", [("/index", {"documents": ["d"]}), ("/query", {"query": "d"})]
)
def test_an_unconfigured_server_refuses_rather_than_running_open(
    monkeypatch: pytest.MonkeyPatch, path: str, body: dict
) -> None:
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    assert client.post(path, json=body).status_code == 503


@pytest.mark.parametrize(
    "path,body", [("/index", {"documents": ["d"]}), ("/query", {"query": "d"})]
)
def test_a_missing_or_wrong_key_is_rejected(
    configured: None, path: str, body: dict
) -> None:
    assert client.post(path, json=body).status_code == 401
    wrong = {"X-API-Key": "x" * len(KEY)}  # same length: not a prefix comparison
    assert client.post(path, json=body, headers=wrong).status_code == 401


def test_the_right_key_is_accepted(configured: None) -> None:
    assert authed.post("/index", json={"documents": ["a doc"]}).status_code == 201
    assert authed.post("/query", json={"query": "doc"}).status_code == 200


def test_health_needs_no_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """A container probe cannot carry a secret, and liveness leaks nothing."""
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    assert client.get("/health").json() == {"status": "ok"}


def test_a_rejected_request_never_touches_the_corpus(configured: None) -> None:
    authed.post("/index", json={"documents": ["the real corpus"]})
    assert client.post("/index", json={"documents": ["hostile"]}).status_code == 401
    body = authed.post("/query", json={"query": "corpus", "k": 1}).json()
    assert body["retrieved"] == ["the real corpus"]


def test_a_body_over_the_bound_is_refused_before_it_is_parsed(
    configured: None,
) -> None:
    oversized = {"documents": ["x" * (DEFAULT_MAX_BODY_BYTES + 1)]}
    assert authed.post("/index", json=oversized).status_code == 413


def test_a_body_under_the_bound_is_accepted(configured: None) -> None:
    ok = {"documents": ["x" * (DEFAULT_MAX_BODY_BYTES // 2)]}
    assert authed.post("/index", json=ok).status_code == 201


def test_the_bound_is_configurable(
    configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", "500")
    assert authed.post("/index", json={"documents": ["x" * 900]}).status_code == 413
    assert authed.post("/index", json={"documents": ["x" * 100]}).status_code == 201


def test_a_post_without_a_length_cannot_stream_past_the_bound(
    configured: None,
) -> None:
    """Content-Length is how the body is bounded before it is read."""
    response = authed.post(
        "/index",
        content=iter([b'{"documents": ["streamed"]}']),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 411


def test_the_bound_does_not_apply_to_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", "1")
    assert client.get("/health").status_code == 200
