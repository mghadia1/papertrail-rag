"""Tests for the Part F keyword strategies and their two documented traps."""

from __future__ import annotations

import pytest

from papertrail.repository import (
    KEYWORD_STRATEGIES,
    RANK_NORMALIZATIONS,
    keyword_search,
    keyword_terms,
    phrase_candidates,
)


def test_keyword_terms_matches_the_documented_tokenizer() -> None:
    # Lower-cased, <3 chars dropped, order-preserving de-duplication, hyphenated
    # tokens kept whole. Note "and" survives: the filter is length-based, not a
    # stop-word list, which is why an all-stop-word AND query is still possible.
    assert keyword_terms("Low-Rank low-rank of a TUNING tuning x1") == (
        "low-rank",
        "tuning",
    )
    assert keyword_terms("a of x") == ()
    assert keyword_terms("the and for") == ("the", "and", "for")


def test_phrase_candidates_finds_quoted_spans_and_capitalised_pairs() -> None:
    assert phrase_candidates('a "chain of thought" prompt') == ["chain of thought"]
    assert phrase_candidates("uses Chain Of Thought here") == [
        "Chain Of",
        "Of Thought",
    ]
    assert phrase_candidates("nothing to see") == []


def test_unsupported_strategy_and_normalization_are_rejected() -> None:
    assert "cascade" in KEYWORD_STRATEGIES
    assert 32 in RANK_NORMALIZATIONS
    with pytest.raises(ValueError, match="unsupported keyword strategy"):
        keyword_search(object(), "q", limit=5, strategy="nope")
    with pytest.raises(ValueError, match="unsupported ts_rank_cd normalization"):
        keyword_search(object(), "q", limit=5, normalization=7)


class _RecordingSession:
    """Captures the tsqueries a strategy issues, without touching a database."""

    def __init__(self, results: list[list[dict]]) -> None:
        self._results = list(results)
        self.calls = 0

    def execute(self, statement):
        self.calls += 1
        rows = self._results.pop(0) if self._results else []

        class _Result:
            def all(self_inner):
                return [
                    type(
                        "Row",
                        (),
                        {
                            "arxiv_id": r["arxiv_id"],
                            "title": "t",
                            "source_url": "u",
                            "chunk_id": r["chunk_id"],
                            "ordinal": 0,
                            "text": "x",
                            "score": r["score"],
                        },
                    )()
                    for r in rows
                ]

        return _Result()


def _row(chunk_id: int, arxiv_id: str, score: float = 1.0) -> dict:
    return {"chunk_id": chunk_id, "arxiv_id": arxiv_id, "score": score}


def test_cascade_falls_through_to_or_when_the_and_branch_is_empty() -> None:
    """Trap: an all-stop-word AND query yields an empty tsquery that matches
    nothing. The cascade must fall through to OR rather than returning nothing."""
    session = _RecordingSession([[], [_row(1, "a1v1"), _row(2, "a2v1")]])
    hits = keyword_search(session, "the and for", limit=5, strategy="cascade")
    assert session.calls == 2  # AND ran, then the OR fill
    assert [h["arxiv_id"] for h in hits] == ["a1v1", "a2v1"]


def test_cascade_keeps_and_rows_ahead_of_or_rows_without_duplicates() -> None:
    session = _RecordingSession(
        [[_row(2, "b1v1")], [_row(1, "a1v1"), _row(2, "b1v1"), _row(3, "c1v1")]]
    )
    hits = keyword_search(session, "alpha beta gamma", limit=5, strategy="cascade")
    # AND row first, then OR rows, and chunk 2 is not repeated.
    assert [h["chunk_id"] for h in hits] == [2, 1, 3]


def test_cascade_short_circuits_when_the_and_branch_fills_the_limit() -> None:
    session = _RecordingSession([[_row(i, f"a{i}v1") for i in range(1, 6)]])
    hits = keyword_search(session, "alpha beta", limit=5, strategy="cascade")
    assert session.calls == 1  # no OR query needed
    assert len(hits) == 5


class _CapturingSession(_RecordingSession):
    """Records the compiled SQL text and bound parameters of each statement."""

    def __init__(self, results):
        super().__init__(results)
        self.sql: list[str] = []
        self.params: list[dict] = []

    def execute(self, statement):
        compiled = statement.compile()
        self.sql.append(str(compiled))
        self.params.append(dict(compiled.params))
        return super().execute(statement)


def test_hyphenated_terms_are_quoted_in_the_and_query() -> None:
    """Trap: an unquoted hyphen can be read as an operator, so each AND term is
    quoted. The AND query text must carry the quoted hyphenated lexeme."""
    session = _CapturingSession([[], []])
    keyword_search(session, "state-of-the-art tuning", limit=5, strategy="cascade")
    and_params = " ".join(str(v) for v in session.params[0].values())
    assert "'state-of-the-art' & 'tuning'" in and_params


def test_cascade_phrase_queries_phrases_before_the_and_branch() -> None:
    session = _CapturingSession([[], [], []])
    keyword_search(
        session, 'a "chain of thought" probe', limit=5, strategy="cascade_phrase"
    )
    assert "phraseto_tsquery" in session.sql[0]
    phrase_params = " ".join(str(v) for v in session.params[0].values())
    assert "chain of thought" in phrase_params
    # The AND branch runs after the phrase branch, and is not itself a phrase query.
    assert "phraseto_tsquery" not in session.sql[1]
    assert "to_tsquery" in session.sql[1]
