"""The corpus is not reachable without a credential, and a body cannot be unbounded.

`/index` replaces the whole corpus and `/query` reads it back, so an open port
was a full read-write handle on the service's data. The bound matters because the
corpus lives in this process: whatever a request carries, the process holds.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import rag_llm_infra.serve as serve
from rag_llm_infra.serve import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_CORPUS_DOCS,
    app,
    require_api_key,
)

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


# The only routes allowed to answer without a credential, and why.
OPEN_ROUTES = {
    "/health": "container probes cannot carry a secret; it returns liveness only",
    "/openapi.json": "the schema, which describes the two guarded routes",
    "/docs": "renders /openapi.json",
    "/docs/oauth2-redirect": "part of the docs UI",
    "/redoc": "renders /openapi.json",
}
GUARDED_ROUTES = {"/index", "/query"}


def _route_paths() -> tuple[set[str], set[str]]:
    """(guarded, open) by dependency identity, not by its name.

    A function whose `__name__` is set to "require_api_key" is not the guard;
    comparing names credits the label instead of the thing.
    """
    guarded: set[str] = set()
    open_paths: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue
        dependencies = getattr(route, "dependencies", [])
        holds = any(d.dependency is require_api_key for d in dependencies)
        (guarded if holds else open_paths).add(path)
    return guarded, open_paths


def test_the_open_routes_are_exactly_these() -> None:
    """The list is the exemption, so widening it must be a visible edit."""
    _, open_paths = _route_paths()
    assert open_paths == set(OPEN_ROUTES)


def test_every_other_route_holds_the_credential_dependency() -> None:
    guarded, _ = _route_paths()
    assert guarded == GUARDED_ROUTES


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


@pytest.mark.parametrize("declared", ["abc", "-1", "1 000", "0x10"])
def test_a_content_length_that_is_not_a_byte_count_is_refused(
    declared: str, configured: None
) -> None:
    """The header is attacker-controlled. `int()` raises on the first three, and
    a negative passes `> limit` and skips the bound: measured 201 before this."""
    body = json.dumps({"documents": ["d"]})
    response = authed.post(
        "/index",
        content=body,
        headers={"content-type": "application/json", "content-length": declared},
    )
    assert response.status_code == 400, response.text


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


@pytest.mark.parametrize(
    "supplied", [b"cl\xc3\xa9", b"\xd0\xba\xd0\xbb\xd1\x8e\xd1\x87", b"x\xff", b"\x80"]
)
def test_a_non_ascii_key_is_rejected_not_a_server_error(
    configured: None, supplied: bytes
) -> None:
    """`compare_digest` raises TypeError on a non-ASCII str, and the header is
    attacker-controlled. A 500 on the credential path is a free traceback.

    Sent as bytes because that is what a socket carries; httpx will not encode a
    non-ASCII str into a header at all, which is why a str probe misses this."""
    response = client.post(
        "/index", json={"documents": ["d"]}, headers={"X-API-Key": supplied}
    )
    assert response.status_code == 401, response.text


def test_a_non_ascii_configured_key_is_a_misconfiguration_not_a_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headers arrive latin-1 while the environment is UTF-8, so such a key
    could never match. Saying so beats rejecting every correct request."""
    monkeypatch.setenv("RAG_API_KEY", "clé-secrète")
    response = client.post("/index", json={"documents": ["d"]})
    assert response.status_code == 503
    assert "ASCII" in response.json()["detail"]


def test_a_whitespace_only_key_is_not_a_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_API_KEY", "   ")
    assert client.post("/index", json={"documents": ["d"]}).status_code == 503


@pytest.mark.parametrize("value", ["abc", "", "-1", "0", "1.5"])
def test_a_malformed_bound_falls_back_instead_of_failing_every_request(
    configured: None, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """It is read per request, so a typo would otherwise be a live outage."""
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", value)
    assert authed.post("/index", json={"documents": ["a doc"]}).status_code == 201


def test_a_corpus_over_the_document_bound_is_refused(configured: None) -> None:
    """The byte bound does not bound memory: each document becomes a
    fixed-width float32 row whatever its length."""
    monkeypatch_free = {"documents": ["d"] * (DEFAULT_MAX_CORPUS_DOCS + 1)}
    assert authed.post("/index", json=monkeypatch_free).status_code == 413


def test_a_corpus_at_the_document_bound_is_accepted(
    configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_MAX_CORPUS_DOCS", "50")
    assert authed.post("/index", json={"documents": ["d"] * 50}).status_code == 201
    assert authed.post("/index", json={"documents": ["d"] * 51}).status_code == 413
