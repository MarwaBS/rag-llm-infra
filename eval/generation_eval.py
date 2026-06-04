"""Generation-quality (faithfulness) gate.

Runs the retrieval step, builds a grounded answer from the retrieved context, and
scores its groundedness — then verifies the metric DISCRIMINATES a grounded answer
from a hallucinated control (same answer + vocabulary absent from the evidence).

Fails CI if the grounded answer scores below threshold or the metric cannot tell a
hallucination apart.

    python -m eval.generation_eval
"""
from __future__ import annotations

import sys
from typing import Dict, List

from rag_llm_infra import get_vector_store, groundedness
from rag_llm_infra._demo import embed

DOCS: List[str] = [
    "FAISS performs in-process vector similarity search with inner product.",
    "Qdrant is a vector database exposing REST and gRPC search APIs.",
    "Retrieval-augmented generation grounds language model output in retrieved documents.",
]
QUERY = "in-process vector similarity search"
GROUNDED_MIN = 0.90
# Content tokens that appear in NO document — a hallucination signature.
OFF_CONTEXT = "bitcoin blockchain quantum supremacy mortgage"


def _retrieve(query: str, k: int = 2) -> List[str]:
    store = get_vector_store("numpy")
    store.add(embed(DOCS))
    _, idx = store.search(embed([query]), k=k)
    return [DOCS[int(i)] for i in idx[0] if i >= 0]


def evaluate() -> Dict[str, float]:
    contexts = _retrieve(QUERY)
    grounded_answer = " ".join(contexts)                         # drawn only from context
    hallucinated_answer = grounded_answer + " " + OFF_CONTEXT     # + unsupported claim
    grounded = groundedness(grounded_answer, contexts)
    hallucinated = groundedness(hallucinated_answer, contexts)
    return {"grounded": grounded, "hallucinated": hallucinated, "margin": grounded - hallucinated}


def main() -> int:
    m = evaluate()
    print(
        f"generation eval — grounded={m['grounded']:.3f}  "
        f"hallucinated={m['hallucinated']:.3f}  margin={m['margin']:.3f}"
    )
    if m["grounded"] < GROUNDED_MIN or m["margin"] <= 0.0:
        print(f"FAIL: grounded < {GROUNDED_MIN} or metric did not discriminate (margin <= 0)")
        return 1
    print("PASS: grounded answer is faithful and the metric flags the hallucination")
    return 0


if __name__ == "__main__":
    sys.exit(main())
