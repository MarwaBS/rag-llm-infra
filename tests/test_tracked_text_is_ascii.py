"""Tracked text is ASCII, apart from one credential fixture.

Ruff, codespell and mypy all pass over a stray em-dash or arrow, so nothing else
reads these bytes. On a cp1252 console U+2192 raises UnicodeEncodeError, so a
tool printing a source line dies on the character rather than the code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIXTURE_WITH_NON_ASCII = "tests/test_serve_auth.py"


def _tracked_files() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [name for name in listed.stdout.split("\n") if name]


def _non_ascii_by_file() -> dict[str, list[tuple[int, str]]]:
    """{path: [(line, character)]}. Undecodable files are binary, so skipped."""
    found: dict[str, list[tuple[int, str]]] = {}
    for name in _tracked_files():
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = [
            (number, character)
            for number, line in enumerate(text.splitlines(), 1)
            for character in line
            if ord(character) > 127
        ]
        if hits:
            found[name] = hits
    return found


def test_the_scan_reaches_the_tree() -> None:
    """A scan returning nothing passes every assertion below."""
    tracked = _tracked_files()
    assert len(tracked) > 50, tracked
    assert FIXTURE_WITH_NON_ASCII in tracked


def test_only_the_credential_fixture_carries_non_ascii() -> None:
    assert set(_non_ascii_by_file()) == {FIXTURE_WITH_NON_ASCII}


def test_the_fixture_still_needs_its_non_ascii_bytes() -> None:
    """Its non-ASCII key proves the credential path answers 503 instead of
    raising TypeError out of compare_digest."""
    assert _non_ascii_by_file().get(FIXTURE_WITH_NON_ASCII)


def test_no_em_dash_survives_anywhere() -> None:
    """The one character that reads as ordinary punctuation in review."""
    carrying = [
        name
        for name, hits in _non_ascii_by_file().items()
        if any(ord(character) == 0x2014 for _, character in hits)
    ]
    assert not carrying
