"""The type gate rejects the shapes it used to accept, over the trees it covers.

`disallow_untyped_defs = true` was satisfied by `Any`: the protocol's core type
was `list[dict[str, Any]]`, so a misspelled key, an integer role and a message
with no role at all type-checked. And `ignore_missing_imports` was global, which
also silences a typo in a first-party import path.

Each case runs mypy over a temporary file. Slow, so they are marked `slow`; they
are the only assertion that the checker's answer changed rather than its config.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Assembled rather than written out, so the spell gate is not asked to make an
# exception for a misspelling that is the point of the fixture.
MISSPELLED = "cont" + "net"

WRONG_SHAPES = [
    (f'[{{"role": "user", "{MISSPELLED}": "typo"}}]', "typeddict-unknown-key"),
    ('[{"role": 42, "content": "int role"}]', "typeddict-item"),
    ('[{"content": "no role"}]', "typeddict-item"),
    ('[{"role": "wizard", "content": "not a role"}]', "typeddict-item"),
]


def _mypy(source: str, tmp_path: Path) -> tuple[int, str]:
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(probe)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return done.returncode, done.stdout + done.stderr


@pytest.mark.slow
@pytest.mark.parametrize("messages,code", WRONG_SHAPES, ids=lambda v: str(v)[:24])
def test_a_wrong_message_shape_is_rejected(
    messages: str, code: str, tmp_path: Path
) -> None:
    source = (
        f"from rag_llm_infra import MockBackend\nMockBackend().invoke({messages})\n"
    )
    returncode, output = _mypy(source, tmp_path)
    assert returncode != 0, output
    assert code in output, output


@pytest.mark.slow
def test_the_control_shows_a_correct_message_is_accepted(tmp_path: Path) -> None:
    source = (
        "from rag_llm_infra import MockBackend\n"
        'MockBackend().invoke([{"role": "user", "content": "fine"}])\n'
    )
    returncode, output = _mypy(source, tmp_path)
    assert returncode == 0, output


@pytest.mark.slow
def test_a_typo_in_a_first_party_import_is_not_swallowed(tmp_path: Path) -> None:
    """This is what a global `ignore_missing_imports` costs."""
    returncode, output = _mypy("import rag_llm_infra.vectorstore\n", tmp_path)
    assert returncode != 0, output
    assert "rag_llm_infra.vectorstore" in output
