"""Tests for the abstention-gate study: AUROC and the gate evidence verifier."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from papertrail.evaluation import auroc, _choose_threshold
from papertrail.evidence import verify_gate_evidence
from papertrail.manifest import CorpusManifest

ROOT = Path(__file__).parents[1]


def test_auroc_known_answers() -> None:
    assert auroc([1.0, 2.0], [-1.0, 0.0]) == 1.0          # perfect separation
    assert auroc([0.0], [1.0]) == 0.0                       # reversed
    assert auroc([1.0], [1.0]) == 0.5                       # single tie
    # pos=[1,2], neg=[2,3]: pairs 1>2 no, 1>3 no, 2>2 tie(.5), 2>3 no → 0.5/4
    assert auroc([1.0, 2.0], [2.0, 3.0]) == pytest.approx(0.125)
    with pytest.raises(ValueError, match="needs at least one"):
        auroc([], [1.0])


def _valid_gate_report(manifest: CorpusManifest) -> dict:
    rows = [
        {"question_id": "q1", "split": "development", "type": "paraphrase", "is_answerable": True, "cos_top": 0.90, "retrieved_rank_of_relevant": 1},
        {"question_id": "q2", "split": "development", "type": "topical", "is_answerable": True, "cos_top": 0.80, "retrieved_rank_of_relevant": 1},
        {"question_id": "n1", "split": "development", "type": "negative_ood", "is_answerable": False, "cos_top": 0.10, "retrieved_rank_of_relevant": None},
        {"question_id": "n2", "split": "development", "type": "negative_near", "is_answerable": False, "cos_top": 0.20, "retrieved_rank_of_relevant": None},
        {"question_id": "q3", "split": "heldout", "type": "paraphrase", "is_answerable": True, "cos_top": 0.85, "retrieved_rank_of_relevant": 1},
        {"question_id": "n3", "split": "heldout", "type": "negative_ood", "is_answerable": False, "cos_top": 0.15, "retrieved_rank_of_relevant": None},
    ]
    pos = [r["cos_top"] for r in rows if r["split"] == "development" and r["is_answerable"]]
    neg = [r["cos_top"] for r in rows if r["split"] == "development" and not r["is_answerable"]]
    ch = _choose_threshold(pos, neg)
    dev = {"cos_top": {
        "auroc": auroc(pos, neg), "threshold": ch["threshold"],
        "dev_accept_rate": ch["development_positive_accept_rate"],
        "dev_abstain_rate": ch["development_negative_abstain_rate"],
        "dev_balanced_accuracy": ch["development_balanced_accuracy"],
    }}
    thr = ch["threshold"]
    pos_h = [r for r in rows if r["split"] == "heldout" and r["is_answerable"]]
    neg_h = [r for r in rows if r["split"] == "heldout" and not r["is_answerable"]]
    accept = statistics.fmean(r["cos_top"] >= thr for r in pos_h)
    abstain = statistics.fmean(r["cos_top"] < thr for r in neg_h)
    heldout = {"cos_top": {
        "false_refusal_rate_answerable": 1.0 - accept,
        "false_answer_rate_ood": statistics.fmean(r["cos_top"] >= thr for r in neg_h),
        "false_answer_rate_near": 0.0,
        "balanced_accuracy": (accept + abstain) / 2,
    }}
    return {
        "created_at_utc": "2026-01-02T00:00:00Z",
        "evaluation_set_frozen_at_utc": "2026-01-01T00:00:00Z",
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "protocol": {"signals": ["cos_top"]},
        "chosen_signal": "cos_top",
        "development": dev,
        "heldout": heldout,
        "rows": rows,
    }


def test_gate_verifier_accepts_valid_and_rejects_edits(tmp_path) -> None:
    manifest = CorpusManifest.read(ROOT / "docs/evidence/corpus-manifest-1000.json")
    report = _valid_gate_report(manifest)
    good = tmp_path / "gate.json"
    good.write_text(json.dumps(report))
    assert verify_gate_evidence(good, manifest)["verified"] is True

    edited = json.loads(good.read_text())
    edited["development"]["cos_top"]["auroc"] = 0.5
    bad = tmp_path / "gate-edit.json"
    bad.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="disagrees"):
        verify_gate_evidence(bad, manifest)

    # a fallback reranker name must be rejected (A6)
    fb = json.loads(good.read_text())
    fb["reranker_model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2-fallback"
    badfb = tmp_path / "gate-fb.json"
    badfb.write_text(json.dumps(fb))
    with pytest.raises(ValueError, match="fallback"):
        verify_gate_evidence(badfb, manifest)
