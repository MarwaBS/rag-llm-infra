"""Derive every eval floor from measurement and write `eval/eval_floors.json`.

    python -m scripts.derive_eval_floors

The gates read the artefact this writes, so no threshold in the repo is a number
somebody picked. `tests/test_eval_floors.py` re-runs this and requires the output
to be byte-identical to the committed file, so a fixture change that moves a
measurement cannot leave a stale floor behind.

Two rules produce the five numbers:

generation — the decision boundary is the midpoint between the worst answer
labelled faithful and the best one labelled hallucinated, and the gate demands at
least half the separation the fixture population actually exhibits.

retrieval — exactly one of the n queries may regress: the floor is (n-1)/n, so
one failure passes and two fail.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from eval import generation_eval, retrieval_eval

from rag_llm_infra import groundedness

ARTEFACT = Path(__file__).resolve().parent.parent / "eval" / "eval_floors.json"


def _floor3(x: float) -> float:
    return math.floor(x * 1000) / 1000


def _generation() -> dict[str, Any]:
    contexts = generation_eval.retrieve(generation_eval.QUERY)
    scored = {
        label: [
            {"answer": a, "score": round(groundedness(a, contexts), 4)} for a in answers
        ]
        for label, answers in (
            ("faithful", generation_eval.FAITHFUL_ANSWERS),
            ("hallucinated", generation_eval.HALLUCINATED_ANSWERS),
        )
    }
    worst_faithful = min(row["score"] for row in scored["faithful"])
    best_hallucinated = max(row["score"] for row in scored["hallucinated"])
    midpoint = (worst_faithful + best_hallucinated) / 2
    return {
        "rule": (
            "boundary = midpoint(worst faithful, best hallucinated); "
            "margin_min = half the observed separation; both floored to 3dp"
        ),
        "measured": {
            **scored,
            "worst_faithful": worst_faithful,
            "best_hallucinated": best_hallucinated,
        },
        "floors": {
            "grounded_min": _floor3(midpoint),
            "hallucinated_max": _floor3(midpoint),
            "margin_min": _floor3((worst_faithful - best_hallucinated) / 2),
        },
    }


def _retrieval() -> dict[str, Any]:
    n = len(retrieval_eval.QUERIES)
    measured = retrieval_eval.evaluate()
    tolerated = _floor3((n - 1) / n)
    return {
        "rule": "floor = (n-1)/n, floored to 3dp: one query may regress, two may not",
        "measured": {"n": n, **{k: round(v, 4) for k, v in measured.items()}},
        "floors": {"recall@1": tolerated, "mrr": tolerated},
    }


def main() -> int:
    payload = {"generation": _generation(), "retrieval": _retrieval()}
    ARTEFACT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {ARTEFACT.name}")
    for section, body in payload.items():
        print(f"  {section}: {body['floors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
