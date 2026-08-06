"""Generation-quality (faithfulness) gate.

Runs the retrieval step, then checks that the groundedness metric DISCRIMINATES
answers labelled faithful (paraphrases of the evidence) from answers labelled
hallucinated (fluent, on-topic, asserting what the evidence does not).

    python -m eval.generation_eval

Both populations are scored. The worst case in each holds the gate. The lowest
faithful score must clear the floor. The highest hallucinated score must stay
under the ceiling. The two must stay a minimum distance apart. Floors come from
`eval/eval_floors.json`, derived by `scripts/derive_eval_floors.py`.

No fixture sits at 0.0 or 1.0. A fixture at a theoretical extreme scores the same
under a metric that discriminates and one that has stopped, so it cannot tell
them apart. `tests/test_generation_eval.py` pins that property.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag_llm_infra import get_vector_store, groundedness
from rag_llm_infra._demo import embed

DOCS: list[str] = [
    "FAISS performs in-process vector similarity search with inner product.",
    "Qdrant is a vector database exposing REST and gRPC search APIs.",
    "Retrieval-augmented generation grounds language model output in retrieved documents.",
]
QUERY = "in-process vector similarity search"

FAITHFUL_ANSWERS: list[str] = [
    "FAISS performs vector similarity search inside the process, ranking by inner product.",
    "Vector similarity search runs in-process in FAISS, scored by inner product.",
    "In-process similarity search over vectors is what FAISS performs, by inner product.",
]
HALLUCINATED_ANSWERS: list[str] = [
    "FAISS uses quantum annealing hardware to accelerate similarity search.",
    "The vector database Redis stores similarity search indexes on tape archives.",
    "Qdrant search runs on a blockchain notary stamped quarterly by astrologers.",
]

_FLOORS = json.loads(
    (Path(__file__).resolve().parent / "eval_floors.json").read_text(encoding="utf-8")
)["generation"]["floors"]
GROUNDED_MIN: float = _FLOORS["grounded_min"]
HALLUCINATED_MAX: float = _FLOORS["hallucinated_max"]
MARGIN_MIN: float = _FLOORS["margin_min"]


def retrieve(query: str, k: int = 2) -> list[str]:
    store = get_vector_store("numpy")
    store.add(embed(DOCS))
    _, idx = store.search(embed([query]), k=k)
    return [DOCS[int(i)] for i in idx[0] if i >= 0]


def evaluate() -> dict[str, float]:
    contexts = retrieve(QUERY)
    grounded = min(groundedness(a, contexts) for a in FAITHFUL_ANSWERS)
    hallucinated = max(groundedness(a, contexts) for a in HALLUCINATED_ANSWERS)
    return {
        "grounded": grounded,
        "hallucinated": hallucinated,
        "margin": grounded - hallucinated,
    }


def main() -> int:
    m = evaluate()
    print(
        f"generation eval — grounded={m['grounded']:.3f}  "
        f"hallucinated={m['hallucinated']:.3f}  margin={m['margin']:.3f}"
    )
    reasons: list[str] = []
    if m["grounded"] < GROUNDED_MIN:
        reasons.append(f"grounded {m['grounded']:.3f} < {GROUNDED_MIN}")
    if m["hallucinated"] > HALLUCINATED_MAX:
        reasons.append(
            f"hallucinated {m['hallucinated']:.3f} > {HALLUCINATED_MAX} (metric did not flag it)"
        )
    if m["margin"] < MARGIN_MIN:
        reasons.append(
            f"margin {m['margin']:.3f} < {MARGIN_MIN} (metric did not discriminate)"
        )
    if reasons:
        print("FAIL: " + "; ".join(reasons))
        return 1
    print(
        "PASS: every faithful answer clears the floor and every hallucination is flagged"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
