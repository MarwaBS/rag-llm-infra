"""The declared floors and bounds, held where they are declared.

`pip-audit` reads the installed environment, so it cannot see a floor deleted
from `pyproject.toml`. It only notices once a resolve happens to land below it.
The size constants have the mirrored problem: the tests that exercise them derive
their payloads from the constant, so any value passes.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from rag_llm_infra.serve import DEFAULT_MAX_BODY_BYTES, DEFAULT_MAX_CORPUS_DOCS

ROOT = Path(__file__).resolve().parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]

# Package -> (floor, the advisory that set it, the extra that pulls it in).
SECURITY_FLOORS = {
    "starlette": ("1.3.1", "PYSEC-2026-248/-249", "serve"),
    "h2": ("4.4.1", "CVE-2026-71554", "qdrant"),
}


def _requirement(extra: str, package: str) -> Requirement | None:
    for line in PROJECT["optional-dependencies"][extra]:
        parsed = Requirement(line)
        if canonicalize_name(parsed.name) == canonicalize_name(package):
            return parsed
    return None


def _assert_floor_holds(extra: str, package: str) -> None:
    """Parse the specifier rather than match its text.

    `starlette>=1.3.1; python_version<'3.0'` contains the floor and applies to no
    interpreter this project admits. A marker makes the constraint conditional,
    so any marker at all disqualifies it.
    """
    floor, advisory, _ = SECURITY_FLOORS[package]
    declared = _requirement(extra, package)
    assert declared, f"{package} floor gone from [{extra}] ({advisory})"
    assert declared.marker is None, f"{declared} is conditional ({advisory})"
    assert declared.specifier.contains(floor), f"{declared} excludes the fixed {floor}"
    fixed = Version(floor)
    below = f"{fixed.major}.{fixed.minor}.{max(fixed.micro - 1, 0)}"
    assert not declared.specifier.contains(below), (
        f"{declared} admits {below}, below the {advisory} fix"
    )


@pytest.mark.parametrize("package", sorted(SECURITY_FLOORS))
def test_the_security_floor_is_declared_where_the_extra_pulls_it(package: str) -> None:
    _assert_floor_holds(SECURITY_FLOORS[package][2], package)


@pytest.mark.parametrize("package", sorted(SECURITY_FLOORS))
def test_the_dev_extra_carries_the_same_floor(package: str) -> None:
    """CI installs `[dev]`, so a floor only in `[serve]` is not what CI audits."""
    _assert_floor_holds("dev", package)


def test_the_body_bound_is_the_documented_one() -> None:
    """`1 MiB` is written in the README, SECURITY.md and the module docstring."""
    assert DEFAULT_MAX_BODY_BYTES == 1024 * 1024


def test_the_corpus_bound_is_the_documented_one() -> None:
    assert DEFAULT_MAX_CORPUS_DOCS == 20_000


def _document_states(document: str, number: str) -> bool:
    """Whether `document` names `number` as a whole number.

    Digit-bounded, because a substring test passes on the wrong value: `2000`
    occurs inside `20000`, and `1 MiB` inside `11 MiB`. A tightened bound would
    otherwise satisfy this against documents still naming the old one.
    """
    text = (ROOT / document).read_text(encoding="utf-8")
    return re.search(rf"(?<!\d){re.escape(number)}(?!\d)", text) is not None


@pytest.mark.parametrize("document", ["README.md", "SECURITY.md"])
def test_the_documents_state_the_body_bound_the_code_uses(document: str) -> None:
    mib = DEFAULT_MAX_BODY_BYTES // (1024 * 1024)
    assert _document_states(document, f"{mib} MiB")


@pytest.mark.parametrize("document", ["README.md", "SECURITY.md"])
def test_the_documents_state_the_corpus_bound_the_code_uses(document: str) -> None:
    """An operator who gets a 413 on an 80 KB body needs the reason written
    somewhere. The byte bound was tied to the docs and this one was not."""
    assert _document_states(document, str(DEFAULT_MAX_CORPUS_DOCS))


def test_every_subprocess_call_is_bounded_by_a_timeout() -> None:
    """An unbounded child hangs the suite or the replay with no exit path.

    Every call site already passes one, so this is a floor rather than a fix:
    the property was hand-verified, and a hand-verified property is one edit
    from being false. Read from the parse tree, because the call can be spelled
    `subprocess.run`, `check_output`, `Popen` or `call`.
    """
    import ast
    import subprocess as sp

    listing = sp.run(
        ["git", "ls-files", "-z", "*.py"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=30,
    )
    spawns = {"subprocess.run", "subprocess.check_output", "subprocess.Popen"}
    unbounded = []
    for name in [n for n in listing.stdout.split("\0") if n]:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in spawns:
                if not any(word.arg == "timeout" for word in node.keywords):
                    unbounded.append(f"{name}:{node.lineno}")
    assert not unbounded, unbounded
