"""Tests for the Phase 1 HNSW recall study: pure recall, the ef_search guard,
and the hnsw evidence verifier."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from papertrail.evaluation import _percentile, chunk_recall
from papertrail.evidence import verify_hnsw_evidence
from papertrail.manifest import CorpusManifest
from papertrail.repository import vector_search

ROOT = Path(__file__).parents[1]


def test_chunk_recall_known_answers() -> None:
    assert chunk_recall([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 5) == 1.0
    # approx returns only 3 rows; exact top-3 all present → 3/3
    assert chunk_recall([1, 2, 3], [1, 2, 3, 4, 5], 3) == 1.0
    # truncated: approx has 2 of the exact top-5, denominator stays 5
    assert chunk_recall([1, 2], [1, 2, 3, 4, 5], 5) == pytest.approx(0.4)
    # disjoint
    assert chunk_recall([9], [1, 2, 3], 3) == 0.0
    with pytest.raises(ValueError, match="k must be positive"):
        chunk_recall([1], [1], 0)


def test_vector_search_rejects_ef_search_below_limit() -> None:
    # The guard raises before any DB access, so no session is needed.
    with pytest.raises(ValueError, match="ef_search must be >= limit"):
        vector_search(None, [0.1, 0.2], limit=50, ef_search=10)
    with pytest.raises(ValueError, match=r"ef_search must be in \[1, 1000\]"):
        vector_search(None, [0.1, 0.2], limit=10, ef_search=0)
    with pytest.raises(ValueError, match=r"ef_search must be in \[1, 1000\]"):
        vector_search(None, [0.1, 0.2], limit=10, ef_search=2000)


def _valid_hnsw_report(manifest: CorpusManifest) -> dict:
    rows = [
        {"question_id": "a", "ef_search": 100, "returned_rows": 50, "truncated": False,
         "recall_at_10": 1.0, "recall_at_50": 1.0, "forced_index_latency_ms": 4.0, "exact_latency_ms": 20.0},
        {"question_id": "b", "ef_search": 100, "returned_rows": 50, "truncated": False,
         "recall_at_10": 0.9, "recall_at_50": 0.8, "forced_index_latency_ms": 6.0, "exact_latency_ms": 24.0},
        {"question_id": "a", "ef_search": 10, "returned_rows": 10, "truncated": True,
         "recall_at_10": 1.0, "recall_at_50": 0.2, "forced_index_latency_ms": 3.0, "exact_latency_ms": 20.0},
        {"question_id": "b", "ef_search": 10, "returned_rows": 10, "truncated": True,
         "recall_at_10": 0.8, "recall_at_50": 0.2, "forced_index_latency_ms": 5.0, "exact_latency_ms": 24.0},
    ]

    def summ(ef, scan):
        sel = [r for r in rows if r["ef_search"] == ef]
        return {
            "questions": len(sel),
            "mean_recall_at_10": statistics.fmean(r["recall_at_10"] for r in sel),
            "mean_recall_at_50": statistics.fmean(r["recall_at_50"] for r in sel),
            "mean_returned_rows": statistics.fmean(r["returned_rows"] for r in sel),
            "forced_index_latency_p50_ms": _percentile([r["forced_index_latency_ms"] for r in sel], 0.50),
            "forced_index_latency_p95_ms": _percentile([r["forced_index_latency_ms"] for r in sel], 0.95),
            "natural_scan": scan,
        }

    exact_vals = list({r["question_id"]: r["exact_latency_ms"] for r in rows}.values())
    return {
        "created_at_utc": "2026-01-02T00:00:00Z",
        "evaluation_set_frozen_at_utc": "2026-01-01T00:00:00Z",
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "protocol": {"limit": 50},
        "rows": rows,
        "summary": {
            "ef_search": {"100": summ(100, "seqscan"), "10": summ(10, "index")},
            "exact": {"latency_p50_ms": _percentile(exact_vals, 0.50),
                      "latency_p95_ms": _percentile(exact_vals, 0.95)},
        },
    }


def test_hnsw_verifier_accepts_valid_and_rejects_edits(tmp_path) -> None:
    manifest = CorpusManifest.read(ROOT / "docs/evidence/corpus-manifest-1000.json")
    report = _valid_hnsw_report(manifest)
    good = tmp_path / "hnsw.json"
    good.write_text(json.dumps(report))
    assert verify_hnsw_evidence(good, manifest)["verified"] is True

    edited = json.loads(good.read_text())
    edited["summary"]["ef_search"]["100"]["mean_recall_at_10"] = 0.5
    bad = tmp_path / "hnsw-edited.json"
    bad.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="disagrees"):
        verify_hnsw_evidence(bad, manifest)

    # a truncated=false row that returned < limit must be rejected
    edited2 = json.loads(good.read_text())
    edited2["rows"][2]["truncated"] = False
    bad2 = tmp_path / "hnsw-trunc.json"
    bad2.write_text(json.dumps(edited2))
    with pytest.raises(ValueError, match="truncated"):
        verify_hnsw_evidence(bad2, manifest)
