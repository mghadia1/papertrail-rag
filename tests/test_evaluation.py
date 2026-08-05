from pathlib import Path
import json

import pytest

from papertrail.evaluation import (
    _choose_threshold,
    load_question_set,
    ndcg_at,
    recall_at,
    reciprocal_rank,
)
from papertrail.manifest import CorpusManifest
from papertrail.evidence import verify_rag_evidence, verify_retrieval_evidence


ROOT = Path(__file__).parents[1]


def test_retrieval_metrics_have_known_answers() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"b"}
    assert recall_at(ranked, relevant, 1) == 0.0
    assert recall_at(ranked, relevant, 5) == 1.0
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert ndcg_at(ranked, relevant, 10) == pytest.approx(1 / 1.584962500721156)


def test_threshold_selection_uses_development_separation() -> None:
    selected = _choose_threshold(
        positive_scores=[0.030, 0.031, 0.032],
        negative_scores=[0.015, 0.016, 0.017],
    )
    assert 0.017 < selected["threshold"] <= 0.030
    assert selected["development_balanced_accuracy"] == 1.0


def test_frozen_question_set_matches_exact_corpus() -> None:
    manifest = CorpusManifest.read(ROOT / "docs/evidence/corpus-manifest-1000.json")
    payload = load_question_set(ROOT / "eval/questions.json", manifest)
    assert len(payload["retrieval_questions"]) == 30
    assert len(payload["abstention_questions"]) == 15
    assert sum(q["split"] == "heldout" for q in payload["retrieval_questions"]) == 10
    v2 = load_question_set(ROOT / "eval/questions-v2.json", manifest)
    assert v2["schema_version"] == 2
    assert len(v2["retrieval_questions"]) == 30
    assert all(
        item["id"].startswith("v2")
        for item in v2["retrieval_questions"]
        if item["split"] == "heldout"
    )


def test_retrieval_verifier_rejects_edited_summary(tmp_path) -> None:
    manifest = CorpusManifest.read(ROOT / "docs/evidence/corpus-manifest-1000.json")
    source = ROOT / "docs/evidence/phase-6-retrieval-evaluation-v2.json"
    assert verify_retrieval_evidence(source, manifest)["verified"] is True
    report = json.loads(source.read_text())
    report["aggregates"]["heldout"]["vector"]["mrr"] = 0.123
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="disagrees"):
        verify_retrieval_evidence(edited, manifest)


def test_rag_verifier_rejects_edited_grounding_rate(tmp_path) -> None:
    manifest = CorpusManifest.read(ROOT / "docs/evidence/corpus-manifest-1000.json")
    questions = load_question_set(ROOT / "eval/questions-v2.json", manifest)
    source = ROOT / "docs/evidence/phase-6-rag-evaluation-v2.json"
    assert verify_rag_evidence(
        source, question_set=questions, expected_threshold=0.03239446668849102
    )["verified"] is True
    report = json.loads(source.read_text())
    report["citation_grounding_rate_among_answers"] = 0.5
    edited = tmp_path / "edited-rag.json"
    edited.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="disagrees"):
        verify_rag_evidence(
            edited, question_set=questions, expected_threshold=0.03239446668849102
        )
