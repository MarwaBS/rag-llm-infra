"""Retrieval-quality eval: recall@1 and MRR on a labelled fixture corpus.

Run as a CI gate:

    python -m eval.retrieval_eval

Exits non-zero if recall@1 or MRR fall below the configured thresholds, so a
retrieval regression fails the build. The corpus uses *paraphrased* queries
(not verbatim documents), so this measures retrieval quality, not identity lookup.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Tuple

from rag_llm_infra import get_vector_store
from rag_llm_infra._demo import embed

# Each query is a paraphrase; the int is the index of its single relevant document.
DOCS: List[str] = [
    "FAISS performs in-process vector similarity search with brute-force inner product.",
    "Qdrant is a vector database exposing REST and gRPC search APIs.",
    "OpenTelemetry standardizes distributed tracing across services.",
    "Retrieval-augmented generation grounds language model output in retrieved documents.",
    "Redis provides an in-memory key-value store with atomic counters and TTL.",
    "Prometheus scrapes time-series metrics from instrumented services.",
]
QUERIES: List[Tuple[str, int]] = [
    ("brute force inner product vector similarity search in process", 0),
    ("vector database exposing grpc and rest search apis", 1),
    ("standardized distributed tracing across services", 2),
    ("grounding language model output in retrieved documents", 3),
    ("in memory key value store with ttl and atomic counters", 4),
    ("scraping time series metrics from instrumented services", 5),
]
THRESHOLDS: Dict[str, float] = {"recall@1": 0.80, "mrr": 0.85}


def evaluate(k: int = 3) -> Dict[str, float]:
    """Return {recall@1, mrr} for the fixture corpus over the NumPy backend."""
    store = get_vector_store("numpy")
    store.add(embed(DOCS))
    hits_at_1 = 0
    rr_sum = 0.0
    for query, gold in QUERIES:
        _, idx = store.search(embed([query]), k=k)
        ranked = [int(i) for i in idx[0] if i >= 0]
        if ranked and ranked[0] == gold:
            hits_at_1 += 1
        for rank, i in enumerate(ranked, start=1):
            if i == gold:
                rr_sum += 1.0 / rank
                break
    n = len(QUERIES)
    return {"recall@1": hits_at_1 / n, "mrr": rr_sum / n}


def main() -> int:
    m = evaluate()
    print(f"retrieval eval — recall@1={m['recall@1']:.3f}  MRR={m['mrr']:.3f}  (n={len(QUERIES)})")
    failures = {k: round(m[k], 3) for k, t in THRESHOLDS.items() if m[k] < t}
    if failures:
        print(f"FAIL: {failures} below thresholds {THRESHOLDS}")
        return 1
    print("PASS: all retrieval metrics meet thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
