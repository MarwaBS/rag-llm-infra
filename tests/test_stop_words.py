"""The stop list is pinned in both directions, and it is wired into the metric.

`_STOP` decides which of an answer's words carry a groundable claim, so every
score depends on its exact membership. It is a judgement call with no derivation
behind it — closed-class English function words — so it is pinned rather than
recomputed. Twenty-seven of its forty entries could be deleted with the whole
suite, both eval gates and the example still green.
"""

import pytest

from rag_llm_infra import groundedness
from rag_llm_infra.faithfulness import _STOP

EXPECTED = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "in", "to", "is", "are",
        "was", "were", "be", "been", "it", "its", "this", "that", "these",
        "those", "with", "for", "on", "by", "as", "from", "at", "into",
        "over", "than", "then", "so", "such", "but", "not", "no", "can",
        "will", "they", "their",
    }
)  # fmt: skip

_UNRELATED = ["cartography of the abyssal seafloor"]


def test_the_stop_list_is_exactly_this() -> None:
    assert _STOP == EXPECTED


@pytest.mark.parametrize("word", sorted(EXPECTED))
def test_a_listed_word_alone_makes_no_claim_to_ground(word: str) -> None:
    # Iterates the pinned copy, not `_STOP`: a removal has to stay in view.
    assert groundedness(word, _UNRELATED) == 1.0


def test_the_control_shows_1_0_is_not_returned_regardless() -> None:
    assert groundedness("bathymetry", _UNRELATED) == 0.0
