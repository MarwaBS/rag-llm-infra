"""SECURITY.md supports the release line that is actually published.

The table named a line that had never shipped, so the only published release
read as unsupported. The version is read from the CHANGELOG, not kept in step.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _latest_released_line() -> str:
    """`major.minor` of the newest dated entry. `unreleased` is skipped:
    publishing starts support, and pyproject's version moves before that."""
    for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        found = re.match(r"##\s*\[(\d+)\.(\d+)\.\d+\]\s*-\s*(?!unreleased)\S", line)
        if found:
            return f"{found.group(1)}.{found.group(2)}"
    raise AssertionError("CHANGELOG.md names no released version")


def _policy_rows() -> dict[str, str]:
    """The support table as {version spec: verdict}. Parsed into rows, because
    `"0.1.x" in policy` also matches `0.1.x is no longer supported`."""
    rows: dict[str, str] = {}
    for line in (ROOT / "SECURITY.md").read_text(encoding="utf-8").splitlines():
        found = re.match(r"\|\s*([<>]?\s*[\d.x]+)\s*\|\s*(\w+)\s*\|", line)
        if found:
            rows[found.group(1).replace(" ", "")] = found.group(2)
    return rows


def test_both_sources_parse() -> None:
    """Either parser returning nothing makes the check below vacuous."""
    assert re.fullmatch(r"\d+\.\d+", _latest_released_line())
    assert len(_policy_rows()) >= 2


def test_the_policy_supports_the_published_line() -> None:
    published = _latest_released_line()
    assert _policy_rows().get(f"{published}.x") == "yes", (
        f"SECURITY.md does not support {published}.x, the newest released line"
    )
