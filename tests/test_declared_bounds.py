"""The declared floors and bounds, held where they are declared.

`pip-audit` reads the installed environment, so it cannot see a floor deleted
from `pyproject.toml`. It only notices once a resolve happens to land below it.
The size constants have the mirrored problem: the tests that exercise them derive
their payloads from the constant, so any value passes.
"""

from __future__ import annotations

import ast
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


def _spellings(tree: ast.Module, exported: set[str]) -> set[str]:
    """How this file can name those members. Read from its own imports, so an
    aliased module or a bare `from subprocess import run` is not missed.

    Seeded with the plain name so the dotted spelling stays covered whether or
    not this file's own import is the one that bound it.
    """
    modules: set[str] = {"subprocess"}
    bare: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "subprocess"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            bare.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in exported
            )
    return {f"{module}.{name}" for module in modules for name in exported} | bare


def _has_deadline(call: ast.Call) -> bool:
    """`timeout=None` restores the unbounded default and `timeout=0` is a
    deadline no child can meet, so neither is a real bound and the
    keyword's presence is not the property wanted.

    A `**kwargs` splat carries no `arg`, so a spawn is flagged even when
    the mapping holds a timeout; write the deadline at the call site.

    A name rather than a literal is taken on trust: `timeout=TIMEOUT_S` is
    read as bound without resolving what `TIMEOUT_S` holds.
    """
    for word in call.keywords:
        if word.arg == "timeout":
            return not (isinstance(word.value, ast.Constant) and not word.value.value)
    return False


def _names_bound_to(tree: ast.Module, spawns: set[str]) -> set[str]:
    """Local names holding a spawn, from a parameter default or an assignment.

    One hop, and simple targets only. A tuple-unpacked target, a name bound
    from another name or a lambda default is not followed: those are
    deliberate spellings, not the drift this floor is here to catch.
    """
    held = (ast.Attribute, ast.Name)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            taking = node.args.posonlyargs + node.args.args
            given = node.args.defaults
            pairs = list(zip(taking[len(taking) - len(given) :], given, strict=True))
            pairs += [
                (arg, default)
                for arg, default in zip(
                    node.args.kwonlyargs, node.args.kw_defaults, strict=True
                )
                if default is not None
            ]
            names.update(
                arg.arg
                for arg, default in pairs
                if isinstance(default, held) and ast.unparse(default) in spawns
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            bound = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node.value, held) and ast.unparse(node.value) in spawns:
                names.update(t.id for t in bound if isinstance(t, ast.Name))
    return names


def test_every_subprocess_call_is_bounded_by_a_timeout() -> None:
    """An unbounded child hangs the suite or the replay with no exit path.

    Of the seven spawns `subprocess` exports, `getoutput`, `getstatusoutput`
    and `Popen` accept no `timeout`, so they are refused rather than waved
    through by a rule they cannot satisfy. A spawn reaching its call site as a
    value is read through the name it is bound to.
    """
    import subprocess as sp

    listing = sp.run(
        ["git", "ls-files", "-z", "*.py"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=True,
        timeout=30,
    )
    takes_timeout = {"run", "call", "check_call", "check_output"}
    takes_none = {"getoutput", "getstatusoutput", "Popen"}
    unbounded: list[str] = []
    refused: list[str] = []
    for name in [n for n in listing.stdout.split("\0") if n]:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        bounded = _spellings(tree, takes_timeout)
        unboundable = _spellings(tree, takes_none)
        aliases = _names_bound_to(tree, bounded)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = ast.unparse(node.func)
                if (callee in bounded or callee in aliases) and not _has_deadline(node):
                    unbounded.append(f"{name}:{node.lineno} {callee}")
            elif isinstance(node, (ast.Name, ast.Attribute)):
                if ast.unparse(node) in unboundable:
                    refused.append(f"{name}:{node.lineno} {ast.unparse(node)}")
    assert not unbounded, unbounded
    assert not refused, refused
