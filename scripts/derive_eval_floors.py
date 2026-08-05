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

retrieval — each floor is the score when exactly one of the n queries slips one
rank. recall@1 counts only rank 1, so it sees a slip and a drop-out alike; MRR
gets its own floor because it is at least recall@1 for every ranking, and sharing
recall's number would leave it unable to reject anything recall had not already.
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
    # Floors derived from the metric cannot also police it: re-deriving after a
    # distortion just moves them out of its way. This is the check that does not
    # move — groundedness is documented as a [0,1] support fraction, so a metric
    # worth gating must put the two labelled populations on opposite sides of
    # that range's midpoint. Below, no floors are written at all.
    if not best_hallucinated < 0.5 <= worst_faithful:
        raise SystemExit(
            f"refusing to derive: the metric does not separate the labelled "
            f"populations across 0.5 (worst faithful {worst_faithful}, "
            f"best hallucinated {best_hallucinated})"
        )
    midpoint = (worst_faithful + best_hallucinated) / 2
    return {
        "rule": (
            "populations must separate across 0.5, the midpoint of the metric's "
            "documented range; boundary = midpoint(worst faithful, best "
            "hallucinated); margin_min = half the observed separation; floored to 3dp"
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
    # One query slipping by a single rank is tolerated; anything worse, or a
    # second one, is not. Each floor is that exact score. MRR needs its own,
    # because MRR >= recall@1 for every ranking: sharing recall's floor would
    # leave it unable to reject anything recall had not rejected first.
    return {
        "rule": (
            "floor = the score when exactly one of n queries slips one rank, "
            "floored to 3dp. mrr rejects a worse slip or a second regression; "
            "recall@1 counts rank 1 only, so it rejects a second one"
        ),
        "measured": {"n": n, **{k: round(v, 4) for k, v in measured.items()}},
        "floors": {
            "recall@1": _floor3((n - 1) / n),
            "mrr": _floor3((n - 1 + 0.5) / n),
        },
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
