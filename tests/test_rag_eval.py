"""Unit test for the D7 two-gate RAG evaluation aggregation (no DB, no Groq)."""

from __future__ import annotations

import json

import papertrail.evaluation as evaluation
from papertrail.evaluation import evaluate_rag
from papertrail.evidence import verify_rag_evidence
from papertrail.generation import AnswerResult


def _result(query, threshold, gate_signal_name):
    common = dict(
        question=query, retrieval_mode="hybrid", threshold=threshold,
        gate_signal_name=gate_signal_name, nearest_papers=(),
    )
    if query == "a":  # answerable, refused, relevant at rank 1
        return AnswerResult(answer=None, abstained=True, abstain_reason="low",
                            top_score=0.01, gate_score=0.01, generator_model=None,
                            citations=(), retrieved_arxiv_ids=("X", "Q"), **common)
    if query == "b":  # answerable, answered and grounded
        return AnswerResult(answer="see [Y]", abstained=False, abstain_reason=None,
                            top_score=0.9, gate_score=0.9, generator_model="g",
                            citations=("Y",), retrieved_arxiv_ids=("Y",), **common)
    if query == "c":  # ood negative, correctly abstains
        return AnswerResult(answer=None, abstained=True, abstain_reason="low",
                            top_score=0.02, gate_score=0.02, generator_model=None,
                            citations=(), retrieved_arxiv_ids=("Z",), **common)
    # query == "d": near-miss negative, wrongly answered (false answer)
    return AnswerResult(answer="claim [W]", abstained=False, abstain_reason=None,
                        top_score=0.8, gate_score=0.8, generator_model="g",
                        citations=("W",), retrieved_arxiv_ids=("W",), **common)


def _question_set():
    return {
        "frozen_at_utc": "2026-01-01T00:00:00Z",
        "schema_version": 3,
        "retrieval_questions": [
            {"id": "h1", "split": "heldout", "type": "paraphrase", "query": "a", "relevant": {"X": 2}},
            {"id": "h2", "split": "heldout", "type": "topical", "query": "b", "relevant": {"Y": 2}},
        ],
        "abstention_questions": [
            {"id": "n1", "split": "heldout", "type": "negative_ood", "query": "c"},
            {"id": "n2", "split": "heldout", "type": "negative_near", "query": "d"},
        ],
    }


class _Enc:
    model_name = "e"


class _Gen:
    model_name = "g"


def test_evaluate_rag_reports_rank_splits(tmp_path, monkeypatch) -> None:
    def fake_answer(session, query, *, encoder, generator, threshold, top_k,
                    gate_signal_name, verify_entailment, min_faithfulness):
        return _result(query, threshold, gate_signal_name)

    monkeypatch.setattr(evaluation, "answer_question", fake_answer)
    report = evaluate_rag(
        object(), question_set=_question_set(), encoder=_Enc(), generator=_Gen(),
        threshold=0.5, output_path=tmp_path / "rag.json", gate_signal_name="rrf_top",
    )
    assert report["gate_signal"] == "rrf_top"
    assert report["heldout_answerable_questions"] == 2
    assert report["answerable_answer_rate"] == 0.5  # only h2 answered
    assert report["answerable_refused_at_rank_1or2"] == 1  # h1 refused, X at rank 1
    assert report["gate_refused_at_rank_1or2"] == 1  # the refusal is a gate refusal
    assert report["entailment_refused_at_rank_1or2"] == 0
    assert report["answerable_uncited_rate"] == 0.0  # no citation-format failures
    assert report["entailment_judge"] is None  # verify_entailment not requested
    assert report["negative_ood_abstain_rate"] == 1.0  # n1 abstained
    assert report["negative_near_abstain_rate"] == 0.0  # n2 wrongly answered
    assert report["entailment_refusals"] == 0
    # written file round-trips
    assert json.loads((tmp_path / "rag.json").read_text())["gate_signal"] == "rrf_top"


def test_evaluate_rag_splits_refusal_causes_and_counts_uncited(tmp_path, monkeypatch) -> None:
    def fake_answer(session, query, *, encoder, generator, threshold, top_k,
                    gate_signal_name, verify_entailment, min_faithfulness):
        if query == "u":  # gpt-oss-style: omitted the [id] format, enforcement raises
            raise ValueError("generated answer contains no versioned arXiv citation")
        if query == "e":  # passed the gate, refused by the faithfulness heuristic
            return AnswerResult(
                question=query, answer=None, abstained=True,
                abstain_reason="failed heuristic token-overlap faithfulness check",
                retrieval_mode="hybrid", top_score=0.9, threshold=threshold,
                generator_model="g", citations=(), retrieved_arxiv_ids=("E",),
                gate_signal_name=gate_signal_name, gate_score=0.9,
                entailment_verified=True, faithfulness_score=0.3, nearest_papers=(),
            )
        return AnswerResult(  # ood negative, abstains at the gate
            question=query, answer=None, abstained=True, abstain_reason="low",
            retrieval_mode="hybrid", top_score=0.01, threshold=threshold,
            generator_model=None, citations=(), retrieved_arxiv_ids=("Z",),
            gate_signal_name=gate_signal_name, gate_score=0.01, nearest_papers=(),
        )

    qs = {
        "frozen_at_utc": "2026-01-01T00:00:00Z",
        "schema_version": 3,
        "retrieval_questions": [
            {"id": "h1", "split": "heldout", "type": "paraphrase", "query": "u", "relevant": {"U": 2}},
            {"id": "h2", "split": "heldout", "type": "topical", "query": "e", "relevant": {"E": 2}},
        ],
        "abstention_questions": [
            {"id": "n1", "split": "heldout", "type": "negative_ood", "query": "c"},
        ],
    }
    monkeypatch.setattr(evaluation, "answer_question", fake_answer)
    out = tmp_path / "rag.json"
    report = evaluate_rag(
        object(), question_set=qs, encoder=_Enc(), generator=_Gen(),
        threshold=0.5, output_path=out, gate_signal_name="cos_mean_top3",
        verify_entailment=True,
    )
    assert report["answerable_uncited_rate"] == 0.5  # h1 of 2 answerable
    assert report["entailment_refused_at_rank_1or2"] == 1  # h2, E at rank 1
    assert report["gate_refused_at_rank_1or2"] == 0  # the refusal was entailment, not the gate
    assert report["entailment_judge"].startswith("HeuristicOverlapJudge")
    assert report["provider_or_enforcement_errors"] == 1  # the uncited ValueError
    # the new fields must recompute cleanly in the verifier
    assert verify_rag_evidence(out, question_set=qs, expected_threshold=0.5)["verified"] is True
