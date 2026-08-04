"""An import a reader can copy out of a docstring must actually import.

Every docstring in the package is parsed, and each absolute import statement it
quotes is executed — the module, and each name it binds, whether that name is an
attribute or a submodule. Relative imports are excluded: they resolve only inside
the package that wrote them, so there is nothing for a reader to copy.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "rag_llm_infra"

_IMPORT_START = re.compile(r"^\s*(?:import|from)\s")


def _docstrings() -> list[tuple[str, str]]:
    """Every docstring in the package, tagged with the file it came from."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                doc = ast.get_docstring(node)
                if doc:
                    found.append((path.name, doc))
    return found


def _import_nodes(doc: str) -> list[ast.Import | ast.ImportFrom]:
    """Import statements in `doc`, as Python's own parser sees them.

    Parsing rather than pattern-matching is what makes the form irrelevant:
    parenthesised, aliased, star, dotted and multi-line imports all arrive as
    the same two node types.
    """
    lines = doc.splitlines()
    nodes: list[ast.Import | ast.ImportFrom] = []
    i = 0
    while i < len(lines):
        if not _IMPORT_START.match(lines[i]):
            i += 1
            continue
        for span in range(1, min(8, len(lines) - i) + 1):
            block = "\n".join(line.strip() for line in lines[i : i + span])
            try:
                body = ast.parse(block).body
            except SyntaxError:
                continue
            if body and all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in body):
                nodes.extend(body)  # type: ignore[arg-type]
                i += span
                break
        else:
            i += 1
    return nodes


def _quoted_imports() -> list[tuple[str, str]]:
    """Every import statement quoted inside a package docstring, unparsed."""
    return [
        (filename, ast.unparse(node))
        for filename, doc in _docstrings()
        for node in _import_nodes(doc)
        # A relative import means nothing outside the package it is written in,
        # so a reader cannot copy it and there is nothing to resolve.
        if not (isinstance(node, ast.ImportFrom) and node.level)
    ]


def _resolve(statement: str) -> None:
    """Import what `statement` imports, including the names it binds."""
    node = ast.parse(statement).body[0]
    if isinstance(node, ast.Import):
        for alias in node.names:
            importlib.import_module(alias.name)
        return
    assert isinstance(node, ast.ImportFrom) and node.module
    module = importlib.import_module(node.module)
    for alias in node.names:
        if alias.name == "*":
            continue
        if hasattr(module, alias.name):
            continue
        # Not an attribute, so it has to be a submodule or the name is wrong.
        importlib.import_module(f"{node.module}.{alias.name}")


def test_the_package_quotes_at_least_one_import() -> None:
    # Without this the parametrized test below would pass on an empty set.
    assert _quoted_imports(), "no import lines found to check — the matcher is dead"


@pytest.mark.parametrize("filename,statement", _quoted_imports())
def test_every_import_shown_in_a_docstring_resolves(
    filename: str, statement: str
) -> None:
    try:
        _resolve(statement)
    except ImportError as exc:
        pytest.fail(f"{filename} shows `{statement}`, which does not import: {exc}")


@pytest.mark.parametrize(
    "statement",
    [
        "import rag_llm_infra.no_such_module",
        "import rag_llm_infra.tracing as t, rag_llm_infra.no_such_module as n",
        "from rag_llm_infra import no_such_name",
        "from rag_llm_infra import (get_llm,\n    no_such_name)",
        "from rag_llm_infra.no_such_module import anything",
    ],
)
def test_resolution_rejects_a_name_or_submodule_that_does_not_exist(
    statement: str,
) -> None:
    with pytest.raises(ImportError):
        _resolve(statement)
