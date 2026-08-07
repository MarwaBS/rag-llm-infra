# Contributing

## Run what CI runs

CI has no step that is not in this list. Running these locally is the whole
check; there is nothing that only happens on the server.

```bash
pip install -e ".[dev]"

ruff check .
ruff format --check .
mypy src eval tests
mypy scripts benchmarks
codespell
pip-audit --strict
pytest -q --cov=src/rag_llm_infra --cov=scripts --cov-fail-under=90
python -m eval.retrieval_eval
python -m eval.generation_eval
python example.py
python -m build
python -m scripts.replay_mutations
```

The one exception is the container job (`docker build`, then start it and call
the API). It needs a container runtime.

`pre-commit install` runs the two ruff commands on commit. A test fails the build
if the hooks and CI drift apart.

## The mutation registry

`tests/mutations.json` holds a defect per guard. `scripts/replay_mutations.py`
applies each one, runs the gates that guard should hold, restores the file, and
fails if any defect survives. **A new behavioural guard needs an entry**, or
nothing has shown it can fail.

One entry is a control marked `"expect": "survives"`. It changes nothing any gate
can see, so a runner that reports everything caught fails on it. Do not delete it.

Adding an entry:

1. Write the guard.
2. Plant the defect by hand and watch the named gate go **red**. On a CRLF
   checkout a byte-level replace can match nothing; assert the text changed
   before believing an exit code.
3. Add the entry with the gate that caught it.
4. `python -m scripts.replay_mutations`.

## Thresholds

No eval floor is edited where it is used. `scripts/derive_eval_floors.py` writes
`eval/eval_floors.json` and a test requires re-running the producer to reproduce
it byte for byte. Change the fixtures, re-run the producer, commit both.

## Documentation

Every sentence in `README.md`, `SECURITY.md`, `CHANGELOG.md` and every docstring
is either mechanically proven or states its own limit. Tests read the README's
request bodies and its relative links, the `.env.example` variable list, and the
version in `SECURITY.md`. If a claim cannot be gated, bound it or cut it.
