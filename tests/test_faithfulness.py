"""The groundedness metric, pinned to exact values taken from its definition.

`groundedness` is the fraction of the answer's content tokens present in the
union of the contexts — a content token being a lowercase `[a-z0-9]+` run of at
least two characters that is not a stop word — and 1.0 for an answer with none.

Every expectation below is that fraction worked out from the definition, not
read off a run. `0 < score < 1` is satisfied by a metric that has stopped reading
the evidence at all; an exact fraction is not.
"""

from fractions import Fraction

import pytest

from rag_llm_infra import groundedness

# (answer, contexts, expected) — expected = |content(answer) & support| / |content(answer)|
CASES: list[tuple[str, list[str], Fraction]] = [
    # whole tokens, not prefixes
    ("process product", ["the processor handles products"], Fraction(0)),
    # every context is evidence, not just the first
    ("gamma delta", ["alpha beta", "gamma delta"], Fraction(1)),
    # digits carry claims
    ("the model has 40 layers", ["the model has 12 layers"], Fraction(3, 4)),
    # stop words are not claims
    ("the vector is on the search", ["vector search"], Fraction(1)),
    # two characters are content, one is not
    ("AI ML", ["the AI and ML pipeline"], Fraction(1)),
    ("AI ML", ["the cooking recipe"], Fraction(0)),
    ("a b c", ["nothing in common here"], Fraction(1)),
    ("", ["anything"], Fraction(1)),
    ("the a an of", ["anything"], Fraction(1)),
    ("unsupported novel claim", [], Fraction(0)),
    # the documented blind spots, as numbers
    ("Paris is not the capital", ["Paris is the capital."], Fraction(1)),
    (
        "The Eiffel Tower is a wrought iron lattice tower in Paris and on Mars",
        ["The Eiffel Tower is a wrought iron lattice tower located in Paris."],
        Fraction(6, 7),
    ),
]


@pytest.mark.parametrize("answer,contexts,expected", CASES)
def test_groundedness_equals_the_fraction_its_definition_gives(
    answer: str, contexts: list[str], expected: Fraction
) -> None:
    assert groundedness(answer, contexts) == pytest.approx(float(expected))


def test_every_case_is_bounded() -> None:
    for answer, contexts, _ in CASES:
        assert 0.0 <= groundedness(answer, contexts) <= 1.0


def test_the_table_covers_both_extremes_and_the_interior() -> None:
    # A table of only 0s and 1s cannot detect a metric that has stopped grading.
    values = {expected for _, _, expected in CASES}
    assert {Fraction(0), Fraction(1)} <= values
    assert any(0 < v < 1 for v in values)
