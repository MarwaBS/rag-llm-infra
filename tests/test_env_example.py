"""`.env.example` documents every variable the package reads.

The list is derived from the source, not kept by hand: `RAG_API_KEY` became
required in 0.2.0 and a reader following `.env.example` would have got 503 with
no explanation.

Bounded: this finds names written as literals that reach `os.getenv`, directly or
through a helper that forwards one of its parameters. A name assembled at runtime
would be missed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _forwarding_parameters(tree: ast.Module) -> dict[str, int]:
    """{helper name: which parameter it hands to os.getenv}."""
    forwarders: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        names = [a.arg for a in node.args.args]
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "getenv"
                and inner.args
                and isinstance(inner.args[0], ast.Name)
                and inner.args[0].id in names
            ):
                forwarders[node.name] = names.index(inner.args[0].id)
    return forwarders


def _read_variables() -> dict[str, str]:
    """{variable: the module that reads it}."""
    found: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forwarders = _forwarding_parameters(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            index = None
            if isinstance(node.func, ast.Attribute) and node.func.attr == "getenv":
                index = 0
            elif isinstance(node.func, ast.Name) and node.func.id in forwarders:
                index = forwarders[node.func.id]
            if index is None or len(node.args) <= index:
                continue
            argument = node.args[index]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                found[argument.value] = path.name
    return found


def test_the_sweep_finds_the_variables_it_is_meant_to_check() -> None:
    """Non-vacuity, and it covers the helper path: `RAG_MAX_BODY_BYTES` is never
    written next to `getenv`, so a literal search for that call misses it."""
    read = _read_variables()
    assert {"RAG_API_KEY", "RAG_MAX_BODY_BYTES", "ENV", "QDRANT_URL"} <= set(read)
    assert "NFKC" not in read, "a non-env literal leaked into the sweep"


@pytest.mark.parametrize("variable", sorted(_read_variables()))
def test_every_variable_the_package_reads_is_documented(variable: str) -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert variable in example, f"{variable} is read by {_read_variables()[variable]}"
