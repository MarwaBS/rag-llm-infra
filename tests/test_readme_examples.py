"""The README's request bodies and relative links are executed, not decorative.

A reader copy-pastes them. Nothing else in the suite reads the README. A payload
that stops matching the request models, or a link to a renamed file, would
otherwise ship without a gate noticing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")

CURL = re.compile(r"curl -XPOST localhost:8000(/\w+) -d '(\{.*?\})'")
LINK = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)")


def _payloads() -> list[tuple[str, dict]]:
    return [(m.group(1), json.loads(m.group(2))) for m in CURL.finditer(README)]


def test_the_curl_bodies_were_found_at_all() -> None:
    paths = [path for path, _ in _payloads()]
    assert paths == ["/index", "/query"], paths


def test_every_documented_call_carries_the_credential() -> None:
    """A copy-pasteable request without the key would 401, and a reader would
    conclude the service is broken rather than that they skipped a step."""
    posts = [ln for ln in README.splitlines() if "curl -XPOST" in ln]
    assert posts
    assert not [ln for ln in posts if "X-API-Key" not in ln]


@pytest.mark.parametrize("path,body", _payloads(), ids=lambda v: str(v)[:20])
def test_each_documented_request_body_is_accepted(
    path: str, body: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag_llm_infra import serve

    monkeypatch.setenv("RAG_API_KEY", "readme-key")
    headers = {"X-API-Key": "readme-key"}
    with TestClient(serve.app, headers=headers) as client:
        # /query needs a corpus, and the README's own /index call is what supplies it.
        for index_path, index_body in _payloads():
            if index_path == "/index":
                client.post(index_path, json=index_body)
        response = client.post(path, json=body)
    assert response.status_code < 300, response.text


def test_every_relative_link_resolves() -> None:
    targets = {m.group(1).split("#")[0] for m in LINK.finditer(README)}
    assert targets, "no relative link found in the README"
    missing = sorted(t for t in targets if t and not (REPO / t).exists())
    assert not missing, f"README links to missing paths: {missing}"
