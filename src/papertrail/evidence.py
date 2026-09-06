"""Independent recomputation checks for PaperTrail evaluation artifacts."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from .evaluation import (
    LEGACY_MODES,
    _aggregate,
    _aggregate_by_type,
    _choose_threshold,
    _percentile,
    auroc,
    chunk_recall,
    ndcg_at,
    recall_at,
    reciprocal_rank,
)

# Tie-break ordering for the gate selection rule (must match score_gate.py).
_GATE_SIGNAL_COST = {"cos_top": 0, "cos_margin": 0, "cos_mean_top3": 0,
                     "kw_top": 1, "rrf_top": 1,
                     "ce_top": 2, "ce_margin": 2, "ce_sigmoid_top": 2}
from .generation import cited_arxiv_ids
from .manifest import CorpusManifest


def _close(actual: float, published: Any, field: str) -> None:
    if not math.isclose(actual, float(published), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"published {field} disagrees with raw evidence")


def _verify_freeze_precedes_report(report: dict[str, Any]) -> None:
    frozen = datetime.fromisoformat(report["evaluation_set_frozen_at_utc"].replace("Z", "+00:00"))
    created = datetime.fromisoformat(report["created_at_utc"].replace("Z", "+00:00"))
    if frozen > created:
        raise ValueError("evaluation freeze timestamp is later than report creation")


def _verify_abstention(report: dict[str, Any]) -> None:
    abstention = report["abstention"]
    selected = _choose_threshold(
        abstention["development_positive_scores"],
        abstention["development_negative_scores"],
    )
    for key, value in selected.items():
        _close(value, abstention[key], f"abstention.{key}")
    frozen = selected["threshold"]
    heldout_positive = statistics.fmean(
        float(score >= frozen) for score in abstention["heldout_positive_scores"]
    )
    heldout_negative = statistics.fmean(
        float(score < frozen) for score in abstention["heldout_negative_scores"]
    )
    _close(heldout_positive, abstention["heldout_positive_accept_rate"], "heldout positive accept rate")
    _close(heldout_negative, abstention["heldout_negative_abstain_rate"], "heldout negative abstain rate")
    _close((heldout_positive + heldout_negative) / 2, abstention["heldout_balanced_accuracy"], "heldout balanced accuracy")


def _verify_retrieval_evidence_v3(
    report: dict[str, Any], question_set: dict[str, Any] | None
) -> dict[str, Any]:
    rows = report.get("per_question", [])
    if not rows:
        raise ValueError("schema-3 retrieval evidence has no raw rows")
    modes = sorted({row["mode"] for row in rows})

    # 1. Recompute every per-row metric from ranked ids + graded relevance.
    for row in rows:
        graded = {str(k): int(v) for k, v in row["relevant"].items()}
        ranked = [str(identifier) for identifier in row["ranked_arxiv_ids"]]
        tag = f"{row['question_id']}/{row['mode']}"
        _close(recall_at(ranked, graded, 5), row["recall_at_5"], f"{tag} recall_at_5")
        _close(recall_at(ranked, graded, 10), row["recall_at_10"], f"{tag} recall_at_10")
        _close(reciprocal_rank(ranked, graded), row["reciprocal_rank"], f"{tag} mrr")
        _close(ndcg_at(ranked, graded, 10), row["ndcg_at_10"], f"{tag} ndcg_at_10")

    # 2. If the question set is supplied, derive the expected coverage from it
    #    (replaces the hard-coded 90/20/10) and check the pooled-topical rows only
    #    ranked ids that were actually judged.
    if question_set is not None:
        for split in ("development", "heldout"):
            expected_ids = sorted(
                q["id"]
                for q in question_set["retrieval_questions"]
                if q["split"] == split
            )
            for mode in modes:
                got = sorted(
                    row["question_id"]
                    for row in rows
                    if row["split"] == split and row["mode"] == mode
                )
                if got != expected_ids:
                    raise ValueError(
                        f"{split}/{mode} rows do not match the question set ids"
                    )
        pools = {
            q["id"]: set(q["pool"])
            for q in question_set["retrieval_questions"]
            if q.get("pool")
        }
        for row in rows:
            pool = pools.get(row["question_id"])
            if pool is not None and not set(row["ranked_arxiv_ids"]).issubset(pool):
                raise ValueError(
                    f"ranked ids for {row['question_id']} fall outside its judged pool"
                )

    # 3. Recompute the per-type and "all" aggregates and check the published block.
    for split in ("development", "heldout"):
        for mode in modes:
            slice_rows = [
                row for row in rows if row["split"] == split and row["mode"] == mode
            ]
            if not slice_rows:
                continue
            published = report["aggregates"][split][mode]
            for type_key, metrics in _aggregate_by_type(slice_rows).items():
                if type_key not in published:
                    raise ValueError(
                        f"aggregates.{split}.{mode} is missing type '{type_key}'"
                    )
                for key, value in metrics.items():
                    _close(
                        float(value),
                        published[type_key][key],
                        f"aggregates.{split}.{mode}.{type_key}.{key}",
                    )

    _verify_abstention(report)
    if report.get("protocol", {}).get("rrf_k") != 60:
        raise ValueError("retrieval evidence does not use frozen RRF k=60")
    return {"verified": True, "kind": "retrieval", "raw_rows": len(rows), "modes": modes}


def verify_retrieval_evidence(
    path: Path,
    manifest: CorpusManifest,
    *,
    question_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    _verify_freeze_precedes_report(report)
    if report.get("corpus_arxiv_ids_sha256") != manifest.arxiv_ids_sha256:
        raise ValueError("retrieval evidence corpus hash does not match manifest")
    if int(report.get("evaluation_schema_version", 0)) >= 3:
        return _verify_retrieval_evidence_v3(report, question_set)

    # Schema 1/2: the original fixed-shape known-item check, preserved exactly.
    rows = report.get("per_question", [])
    if len(rows) != 90:
        raise ValueError(f"retrieval evidence must contain 90 raw rows; found {len(rows)}")
    for split, expected in (("development", 20), ("heldout", 10)):
        for mode in LEGACY_MODES:
            selected = [row for row in rows if row["split"] == split and row["mode"] == mode]
            if len(selected) != expected:
                raise ValueError(f"expected {expected} {split}/{mode} rows; found {len(selected)}")
            calculated = _aggregate(selected)
            published = report["aggregates"][split][mode]
            for key, value in calculated.items():
                _close(float(value), published[key], f"aggregates.{split}.{mode}.{key}")

    _verify_abstention(report)
    if report.get("protocol", {}).get("rrf_k") != 60:
        raise ValueError("retrieval evidence does not use frozen RRF k=60")
    return {"verified": True, "kind": "retrieval", "raw_rows": len(rows)}


def verify_hnsw_evidence(path: Path, manifest: CorpusManifest) -> dict[str, Any]:
    """Recompute the HNSW recall study's summary from its raw rows."""
    report = json.loads(path.read_text(encoding="utf-8"))
    _verify_freeze_precedes_report(report)
    if report.get("corpus_arxiv_ids_sha256") != manifest.arxiv_ids_sha256:
        raise ValueError("hnsw evidence corpus hash does not match manifest")
    rows = report.get("rows", [])
    if not rows:
        raise ValueError("hnsw evidence has no rows")
    limit = int(report["protocol"]["limit"])

    # Truncation honesty (C3): ef_search < limit must return < limit rows and be
    # flagged truncated; the flag must match returned_rows.
    for row in rows:
        truncated = bool(row["returned_rows"] < limit)
        if truncated != bool(row["truncated"]):
            raise ValueError(f"truncated flag disagrees with returned_rows for {row['question_id']}/{row['ef_search']}")
        if int(row["ef_search"]) < limit and not row["truncated"]:
            raise ValueError(f"ef_search<{limit} must be truncated for {row['question_id']}")

    summary = report["summary"]["ef_search"]
    for ef_str, published in summary.items():
        sel = [r for r in rows if r["ef_search"] == int(ef_str)]
        _close(len(sel), published["questions"], f"ef {ef_str} questions")
        _close(statistics.fmean(r["recall_at_10"] for r in sel), published["mean_recall_at_10"], f"ef {ef_str} recall@10")
        _close(statistics.fmean(r["recall_at_50"] for r in sel), published["mean_recall_at_50"], f"ef {ef_str} recall@50")
        _close(statistics.fmean(r["returned_rows"] for r in sel), published["mean_returned_rows"], f"ef {ef_str} returned")
        _close(_percentile([r["forced_index_latency_ms"] for r in sel], 0.50), published["forced_index_latency_p50_ms"], f"ef {ef_str} p50")
        _close(_percentile([r["forced_index_latency_ms"] for r in sel], 0.95), published["forced_index_latency_p95_ms"], f"ef {ef_str} p95")
        if published.get("natural_scan") not in ("index", "seqscan"):
            raise ValueError(f"ef {ef_str} natural_scan must be 'index' or 'seqscan'")

    # Exact latency percentiles recompute from the distinct per-question value.
    exact_vals = list({r["question_id"]: r["exact_latency_ms"] for r in rows}.values())
    _close(_percentile(exact_vals, 0.50), report["summary"]["exact"]["latency_p50_ms"], "exact p50")
    _close(_percentile(exact_vals, 0.95), report["summary"]["exact"]["latency_p95_ms"], "exact p95")
    return {"verified": True, "kind": "hnsw", "rows": len(rows)}


def verify_gate_evidence(path: Path, manifest: CorpusManifest) -> dict[str, Any]:
    """Recompute the abstention-gate selection (D2/D3) from its embedded rows."""
    report = json.loads(path.read_text(encoding="utf-8"))
    _verify_freeze_precedes_report(report)
    if report.get("corpus_arxiv_ids_sha256") != manifest.arxiv_ids_sha256:
        raise ValueError("gate evidence corpus hash does not match manifest")
    model = report.get("reranker_model", "")
    if not model or model.endswith("-fallback"):
        raise ValueError("gate evidence reranker model missing or a fallback (A6)")
    rows = report.get("rows", [])
    signals = report["protocol"]["signals"]
    dev = report["development"]

    for sig in signals:
        pos = [r[sig] for r in rows if r["split"] == "development" and r["is_answerable"]]
        neg = [r[sig] for r in rows if r["split"] == "development" and not r["is_answerable"]]
        _close(auroc(pos, neg), dev[sig]["auroc"], f"dev auroc {sig}")
        chosen = _choose_threshold(pos, neg)
        _close(chosen["threshold"], dev[sig]["threshold"], f"dev threshold {sig}")
        _close(chosen["development_balanced_accuracy"], dev[sig]["dev_balanced_accuracy"], f"dev balacc {sig}")

    expected = sorted(
        signals,
        key=lambda s: (-dev[s]["auroc"], -dev[s]["dev_balanced_accuracy"], _GATE_SIGNAL_COST.get(s, 9), s),
    )[0]
    if report["chosen_signal"] != expected:
        raise ValueError(f"chosen_signal {report['chosen_signal']!r} does not follow the selection rule (expected {expected!r})")

    held = [r for r in rows if r["split"] == "heldout"]
    pos_h = [r for r in held if r["is_answerable"]]
    neg_h = [r for r in held if not r["is_answerable"]]
    for sig in signals:
        thr = dev[sig]["threshold"]
        accept = statistics.fmean(r[sig] >= thr for r in pos_h)
        abstain = statistics.fmean(r[sig] < thr for r in neg_h)
        _close(1.0 - accept, report["heldout"][sig]["false_refusal_rate_answerable"], f"ho refuse {sig}")
        _close((accept + abstain) / 2, report["heldout"][sig]["balanced_accuracy"], f"ho balacc {sig}")
    return {"verified": True, "kind": "gate", "rows": len(rows), "chosen_signal": report["chosen_signal"]}


def verify_rag_evidence(
    path: Path, *, question_set: dict[str, Any], expected_threshold: float
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    _verify_freeze_precedes_report(report)
    if report.get("evaluation_set_frozen_at_utc") != question_set.get("frozen_at_utc"):
        raise ValueError("RAG evidence freeze timestamp does not match the question set")
    records = report.get("records", [])
    expected_ids = {
        item["id"]
        for key in ("retrieval_questions", "abstention_questions")
        for item in question_set[key]
        if item["split"] == "heldout"
    }
    # Count derived from the question set (was hard-coded 15 for v2's held-out;
    # v3 held-out is larger).
    if len(records) != len(expected_ids):
        raise ValueError(
            f"RAG evidence must contain {len(expected_ids)} records; found {len(records)}"
        )
    actual_ids = [row["question_id"] for row in records]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("RAG evidence question IDs do not match the frozen held-out set")
    _close(expected_threshold, report["frozen_abstain_threshold"], "frozen abstain threshold")

    for row in records:
        parsed = list(cited_arxiv_ids(row["answer"] or ""))
        if parsed != row["citations"]:
            raise ValueError(f"published citations disagree with answer for {row['question_id']}")
        if report.get("schema_version") == 1:
            grounded = False if row["error"] else set(parsed).issubset(row["retrieved_arxiv_ids"])
        else:
            grounded = bool(row["answer"] and parsed) and set(parsed).issubset(
                row["retrieved_arxiv_ids"]
            )
        if bool(row["citations_all_retrieved"]) != grounded:
            raise ValueError(f"published grounding flag disagrees for {row['question_id']}")

    # Internal consistency: the gate must have been applied at the declared
    # threshold (only present on schema-3 gate runs, which carry gate_score).
    frozen = report["frozen_abstain_threshold"]
    for row in records:
        gate_score = row.get("gate_score")
        if gate_score is None or row["error"]:
            continue
        if gate_score < frozen and not row["abstained"]:
            raise ValueError(f"{row['question_id']} scored below threshold but was not abstained")
        if gate_score >= frozen and row["abstained"] and not row.get("entailment_refused"):
            raise ValueError(f"{row['question_id']} passed the gate but abstained without an entailment refusal")

    positives = [row for row in records if row["expected_answerable"]]
    negatives = [row for row in records if not row["expected_answerable"]]
    generated = [row for row in positives if not row["abstained"] and row["answer"]]
    calculated = {
        "heldout_answerable_questions": len(positives),
        "heldout_out_of_domain_questions": len(negatives),
        "answerable_answer_rate": statistics.fmean(float(bool(row["answer"])) for row in positives),
        "answerable_abstain_rate": statistics.fmean(float(row["abstained"]) for row in positives),
        "out_of_domain_abstain_rate": statistics.fmean(float(row["abstained"]) for row in negatives),
        "provider_or_enforcement_errors": sum(bool(row["error"]) for row in records),
        "citation_grounding_rate_among_answers": (
            statistics.fmean(float(row["citations_all_retrieved"]) for row in generated)
            if generated else 0.0
        ),
    }
    for key, value in calculated.items():
        _close(float(value), report[key], key)

    # D7 metrics, present only on the two-gate evaluation files; recompute if present.
    if "answerable_refused_at_rank_1or2" in report:
        _close(
            sum(row["abstained"] and row.get("retrieved_rank_of_relevant") in (1, 2) for row in positives),
            report["answerable_refused_at_rank_1or2"], "answerable_refused_at_rank_1or2",
        )
    if "entailment_refusals" in report:
        _close(
            sum(bool(row.get("entailment_refused")) for row in records),
            report["entailment_refusals"], "entailment_refusals",
        )
    for split_key, subset in (
        ("negative_ood_abstain_rate", [r for r in negatives if r.get("type") == "negative_ood"]),
        ("negative_near_abstain_rate", [r for r in negatives if r.get("type") == "negative_near"]),
    ):
        if split_key in report:
            recomputed = statistics.fmean(float(r["abstained"]) for r in subset) if subset else 0.0
            _close(recomputed, report[split_key], split_key)
    return {"verified": True, "kind": "rag", "raw_records": len(records)}
