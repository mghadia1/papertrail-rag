"""Independent recomputation checks for PaperTrail evaluation artifacts."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from .evaluation import MODES, _aggregate, _choose_threshold
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


def verify_retrieval_evidence(path: Path, manifest: CorpusManifest) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    _verify_freeze_precedes_report(report)
    if report.get("corpus_arxiv_ids_sha256") != manifest.arxiv_ids_sha256:
        raise ValueError("retrieval evidence corpus hash does not match manifest")
    rows = report.get("per_question", [])
    if len(rows) != 90:
        raise ValueError(f"retrieval evidence must contain 90 raw rows; found {len(rows)}")
    for split, expected in (("development", 20), ("heldout", 10)):
        for mode in MODES:
            selected = [row for row in rows if row["split"] == split and row["mode"] == mode]
            if len(selected) != expected:
                raise ValueError(f"expected {expected} {split}/{mode} rows; found {len(selected)}")
            calculated = _aggregate(selected)
            published = report["aggregates"][split][mode]
            for key, value in calculated.items():
                _close(float(value), published[key], f"aggregates.{split}.{mode}.{key}")

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
    if report.get("protocol", {}).get("rrf_k") != 60:
        raise ValueError("retrieval evidence does not use frozen RRF k=60")
    return {"verified": True, "kind": "retrieval", "raw_rows": len(rows)}


def verify_rag_evidence(
    path: Path, *, question_set: dict[str, Any], expected_threshold: float
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    _verify_freeze_precedes_report(report)
    if report.get("evaluation_set_frozen_at_utc") != question_set.get("frozen_at_utc"):
        raise ValueError("RAG evidence freeze timestamp does not match the question set")
    records = report.get("records", [])
    if len(records) != 15:
        raise ValueError(f"RAG evidence must contain 15 raw records; found {len(records)}")
    expected_ids = {
        item["id"]
        for key in ("retrieval_questions", "abstention_questions")
        for item in question_set[key]
        if item["split"] == "heldout"
    }
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
    return {"verified": True, "kind": "rag", "raw_records": len(records)}
