from pathlib import Path
import json
import math

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


def test_graded_ndcg_has_known_answer() -> None:
    # grades a=2, b=1 ranked as [b, a]:
    #   DCG  = (2**1-1)/log2(2) + (2**2-1)/log2(3) = 1 + 3/log2(3)
    #   IDCG = (2**2-1)/log2(2) + (2**1-1)/log2(3) = 3 + 1/log2(3)
    dcg = 1.0 + 3.0 / math.log2(3)
    idcg = 3.0 + 1.0 / math.log2(3)
    assert ndcg_at(["b", "a"], {"a": 2, "b": 1}, 10) == pytest.approx(dcg / idcg)
    # A binary set reduces to the original binary nDCG (grade 1 gain == 1).
    assert ndcg_at(["b", "a"], {"a", "b"}, 10) == pytest.approx(1.0)


def test_schema3_loader_round_trips_graded_relevance(tmp_path) -> None:
    manifest = CorpusManifest.read(ROOT / "docs/evidence/corpus-manifest-1000.json")
    first, second = manifest.arxiv_ids[0], manifest.arxiv_ids[1]
    payload = {
        "schema_version": 3,
        "frozen_at_utc": "2026-09-05T00:00:00Z",
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "sampling_seed": 20260903,
        "retrieval_questions": [
            {
                "id": "v3q001",
                "split": "development",
                "type": "paraphrase",
                "query": "a made-up query",
                "relevant": {first: 2},
            },
            {
                "id": "v3q002",
                "split": "heldout",
                "type": "topical",
                "query": "another query",
                "relevant": {first: 2, second: 1},
                "pool": [first, second],
            },
        ],
        "abstention_questions": [
            {
                "id": "v3n001",
                "split": "development",
                "type": "negative_near",
                "query": "an absent topic",
            }
        ],
    }
    path = tmp_path / "questions-v3-tiny.json"
    path.write_text(json.dumps(payload))
    loaded = load_question_set(path, manifest)
    assert loaded["schema_version"] == 3
    first_q = loaded["retrieval_questions"][0]
    assert first_q["relevant"] == {first: 2}
    assert first_q["relevant_arxiv_ids"] == [first]
    assert all(
        isinstance(grade, int)
        for item in loaded["retrieval_questions"]
        for grade in item["relevant"].values()
    )

    # An out-of-range grade is rejected.
    payload["retrieval_questions"][0]["relevant"] = {first: 3}
    bad = tmp_path / "questions-v3-badgrade.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="grade"):
        load_question_set(bad, manifest)


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


def test_rerank_verifier_requires_reranker_protocol_fields(tmp_path) -> None:
    manifest = CorpusManifest.read(ROOT / "docs/evidence/corpus-manifest-1000.json")
    questions = load_question_set(ROOT / "eval/questions-v3.json", manifest)
    source = ROOT / "docs/evidence/phase-8-rerank-msmarco-pool20-hybrid_rerank.json"
    assert verify_retrieval_evidence(source, manifest, question_set=questions)[
        "verified"
    ] is True

    # E5: a new-shape rerank file (per-row rerank_pool_size present) must carry a
    # non-fallback reranker model and pool depth in its protocol.
    report = json.loads(source.read_text())
    dropped = json.loads(source.read_text())
    dropped["protocol"].pop("reranker_model")
    edited = tmp_path / "no-model.json"
    edited.write_text(json.dumps(dropped))
    with pytest.raises(ValueError, match="reranker_model missing"):
        verify_retrieval_evidence(edited, manifest, question_set=questions)

    fallback = json.loads(source.read_text())
    fallback["protocol"]["reranker_model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2-fallback"
    edited2 = tmp_path / "fallback.json"
    edited2.write_text(json.dumps(fallback))
    with pytest.raises(ValueError, match="fallback"):
        verify_retrieval_evidence(edited2, manifest, question_set=questions)

    no_pool = report
    no_pool["protocol"].pop("rerank_pool")
    edited3 = tmp_path / "no-pool.json"
    edited3.write_text(json.dumps(no_pool))
    with pytest.raises(ValueError, match="rerank_pool missing"):
        verify_retrieval_evidence(edited3, manifest, question_set=questions)


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
