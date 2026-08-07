## What changed, and why

<!-- The mechanism, not the effort. -->

## Evidence

<!-- Paste the output, not the claim. Delete rows that do not apply. -->

| Gate | Result |
| --- | --- |
| `pytest --cov-fail-under=90` | |
| `ruff check .` / `ruff format --check .` | |
| `mypy src eval tests` / `mypy scripts benchmarks` | |
| `codespell` / `pip-audit --strict` | |
| `python -m eval.retrieval_eval` / `generation_eval` | |
| `python -m scripts.replay_mutations` | |

## Behavioural change

- [ ] A new guard has a registry entry, and I watched it go **red** before it went green
- [ ] No shipped sentence claims something no gate holds
- [ ] `CHANGELOG.md` records anything a caller can observe

## Not done

<!-- What you left out, and why. An empty section is a claim in itself. -->
