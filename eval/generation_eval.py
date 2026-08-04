"""Generation-quality (faithfulness) gate.

Runs the retrieval step, then checks that the groundedness metric DISCRIMINATES a
faithful answer (drawn from the evidence) from an independent hallucinated answer
(plausible-sounding, but every claim absent from the evidence).

The gate is two-sided: the faithful answer must score above a floor AND the
hallucinated answer below a ceiling, with a minimum margin between them. It fails
if the metric stops separating the two cases (a metric returning any constant
trips either the ceiling or the floor), if retrieval regresses so the faithful
answer is no longer supported, or if the margin collapses.

The floor is the weak side: `FAITHFUL_ANSWER` is `DOCS[0]` verbatim, so it scores
1.0 by identity and the floor cannot discriminate. The ceiling and the margin are
what this gate rests on; `tests/test_faithfulness.py` is what holds the metric
itself.

    python -m eval.generation_eval
"""

from __future__ import annotations

import sys

from rag_llm_infra import get_vector_store, groundedness
from rag_llm_infra._demo import embed

DOCS: list[str] = [
    "FAISS performs in-process vector similarity search with inner product.",
    "Qdrant is a vector database exposing REST and gRPC search APIs.",
    "Retrieval-augmented generation grounds language model output in retrieved documents.",
]
QUERY = "in-process vector similarity search"

# A faithful answer: DOCS[0] verbatim, the unambiguous match for QUERY. See the
# module docstring on why this makes the floor non-discriminating.
FAITHFUL_ANSWER = (
    "FAISS performs in-process vector similarity search with inner product."
)
# A hallucinated answer: fluent, on-topic-sounding, but every content word is
# absent from the evidence. An honest metric must score this near zero.
HALLUCINATED_ANSWER = (
    "Bitcoin blockchain mining reached quantum supremacy to approve mortgage "
    "applications via astrology and homeopathy."
)

GROUNDED_MIN = 0.90  # a faithful answer must be strongly supported
HALLUCINATED_MAX = 0.34  # a hallucination must be clearly flagged as unsupported
MARGIN_MIN = 0.50  # and the two must be well-separated


def _retrieve(query: str, k: int = 2) -> list[str]:
    store = get_vector_store("numpy")
    store.add(embed(DOCS))
    _, idx = store.search(embed([query]), k=k)
    return [DOCS[int(i)] for i in idx[0] if i >= 0]


def evaluate() -> dict[str, float]:
    contexts = _retrieve(QUERY)
    grounded = groundedness(FAITHFUL_ANSWER, contexts)
    hallucinated = groundedness(HALLUCINATED_ANSWER, contexts)
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
        "PASS: faithful answer is supported and the hallucination is flagged below ceiling"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
