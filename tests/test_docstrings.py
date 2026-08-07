"""An import a reader can copy out of a docstring must actually import.

Each absolute import quoted in a package docstring is executed: the module, and
each name it binds, as an attribute or as a submodule. Relative imports resolve
only inside the package that wrote them, so there is nothing to copy.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "rag_llm_infra"

# A quoted import may sit behind a doctest prompt, a shell prompt or a bullet.
_DECORATION = re.compile(r"^\s*(?:>>>|\.\.\.|[-*+]|\$)?\s*")
_IMPORT_START = re.compile(r"^\s*(?:>>>|\.\.\.|[-*+]|\$)?\s*(?:import|from)\s")


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
    """Import statements in `doc`, as Python's own parser sees them."""
    lines = [_DECORATION.sub("", line) for line in doc.splitlines()]
    starts = [bool(_IMPORT_START.match(line)) for line in doc.splitlines()]
    nodes: list[ast.Import | ast.ImportFrom] = []
    i = 0
    while i < len(lines):
        if not starts[i]:
            i += 1
            continue
        # Brackets wrap over any number of lines but not over a blank one.
        end = i
        while end < len(lines) and lines[end].strip():
            end += 1
        for span in range(1, end - i + 1):
            try:
                body = ast.parse("\n".join(lines[i : i + span])).body
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
        if not (isinstance(node, ast.ImportFrom) and node.level)
    ]


def _resolve(statement: str) -> None:
    """Import what `statement` imports, including the names it binds."""
    for node in ast.parse(statement).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                importlib.import_module(alias.name)
            continue
        assert isinstance(node, ast.ImportFrom) and node.module
        module = importlib.import_module(node.module)
        for alias in node.names:
            if alias.name == "*" or hasattr(module, alias.name):
                continue
            # Not an attribute, so it has to be a submodule or the name is wrong.
            importlib.import_module(f"{node.module}.{alias.name}")


def test_the_package_quotes_at_least_one_import() -> None:
    # Without this the parametrized test below would pass on an empty set.
    assert _quoted_imports(), "no import statements collected; the collector is dead"


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
        "import rag_llm_infra.tracing\nimport rag_llm_infra.no_such_module",
    ],
)
def test_resolution_rejects_a_name_or_submodule_that_does_not_exist(
    statement: str,
) -> None:
    with pytest.raises(ImportError):
        _resolve(statement)


@pytest.mark.parametrize(
    "doc,expected",
    [
        ("import os", ["import os"]),
        (">>> import os", ["import os"]),
        ("    - from os import path", ["from os import path"]),
        ("$ import os", ["import os"]),
        (
            "from os import (\n" + "".join(f"    {n},\n" for n in "abcdefghij") + ")",
            ["from os import a, b, c, d, e, f, g, h, i, j"],
        ),
        ("import os  # trailing comment", ["import os"]),
        ("import os\nimport sys", ["import os", "import sys"]),
        ("import os\n\nimport sys", ["import os", "import sys"]),
        # Prose that merely opens with one of the keywords is not a statement.
        ("import the package before use", []),
        ("from the retrieved documents, import order follows", []),
        ("imported lazily so the SDK stays optional", []),
    ],
)
def test_the_collector_finds_a_statement_in_any_form_and_prose_in_none(
    doc: str, expected: list[str]
) -> None:
    assert [ast.unparse(n) for n in _import_nodes(doc)] == expected
