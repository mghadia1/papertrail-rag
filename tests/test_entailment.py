"""Tests for the statement-level token-overlap faithfulness heuristic."""

from __future__ import annotations

from papertrail.entailment import (
    EntailmentVerdict,
    HeuristicOverlapJudge,
    evaluate_entailment,
    extract_statements,
)


def test_extract_statements_splits_sentences_and_captures_citations():
    answer = (
        "FlashAttention uses SRAM tiling to optimize memory access [2205.14135v2]. "
        "It reduces intermediate matrix materialization in high-bandwidth memory [2205.14135v2].\n\n"
        "Another model uses standard multi-head attention."
    )
    statements = extract_statements(answer)
    assert len(statements) == 3
    assert statements[0][1] == "2205.14135v2"
    assert "SRAM tiling" in statements[0][0]
    assert statements[2][1] is None


def test_heuristic_overlap_judge_supports_matching_fact():
    judge = HeuristicOverlapJudge()
    premise = (
        "FlashAttention is a fast and memory-efficient exact attention algorithm with IO-awareness. "
        "It tiles keys and values to compute attention within SRAM without writing large matrices to HBM."
    )
    statement = "FlashAttention computes exact attention within fast SRAM without writing large matrices to HBM."
    verdict = judge.verify_statement(statement, premise, "2205.14135v2")

    assert verdict.verdict == EntailmentVerdict.ENTAILED
    assert verdict.confidence > 0.6


def test_heuristic_overlap_judge_detects_polarity_contradiction():
    judge = HeuristicOverlapJudge()
    premise = "The proposed algorithm completely fails on out-of-distribution tabular benchmarks."
    statement = "The proposed algorithm succeeds on out-of-distribution tabular benchmarks."
    verdict = judge.verify_statement(statement, premise, "2401.99999v1")

    assert verdict.verdict == EntailmentVerdict.CONTRADICTED


def test_evaluate_entailment_abstained_on_contradictions_and_unsupported_claims():
    context_by_id = {
        "2205.14135v2": "FlashAttention uses SRAM tiling to accelerate transformer execution on GPUs."
    }
    answer = (
        "FlashAttention uses SRAM tiling on GPUs [2205.14135v2]. "
        "It also completely solves the Riemann hypothesis in quantum physics [2205.14135v2]."
    )

    report = evaluate_entailment(answer, context_by_id, min_faithfulness_threshold=0.80)

    assert not report.is_faithful
    assert report.total_statements == 2
    assert report.entailed_statements == 1
    assert report.unsupported_statements == 1
    assert report.faithfulness_score == 0.5
    assert len(report.ungrounded_claims) == 1
