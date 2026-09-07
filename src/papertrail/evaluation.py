"""Predeclared retrieval and abstention evaluation for PaperTrail."""

from __future__ import annotations

import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from .embedding import Encoder
from .generation import Generator, answer_question
from .manifest import CorpusManifest
from .retrieval import SearchMode, retrieve


# The go-forward modes evaluated for schema-3 (graded) question sets. Schema 1/2
# sets predate the cross-encoder and are still evaluated over the original three,
# so their frozen evidence keeps verifying unchanged.
MODES: tuple[SearchMode, ...] = ("vector", "keyword", "hybrid", "hybrid_rerank")
LEGACY_MODES: tuple[SearchMode, ...] = ("vector", "keyword", "hybrid")


def _modes_for_schema(schema_version: int) -> tuple[SearchMode, ...]:
    return MODES if schema_version >= 3 else LEGACY_MODES


def _as_grades(relevant: "set[str] | dict[str, int]") -> dict[str, int]:
    """Accept a binary set or a graded map; return a grade map (binary → grade 1)."""
    if isinstance(relevant, dict):
        return {str(k): int(v) for k, v in relevant.items()}
    return {str(identifier): 1 for identifier in relevant}


def auroc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Rank-sum AUROC: P(random positive scores above random negative), ties 0.5.

    Higher scores are assumed more confident (answerable). O(n·m); the gate study
    has at most ~90 positives × ~27 negatives, so the quadratic form is fine.
    """
    if not positive_scores or not negative_scores:
        raise ValueError("auroc needs at least one positive and one negative score")
    wins = 0.0
    for p in positive_scores:
        for n in negative_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


def chunk_recall(approx_ids: list[int], exact_ids: list[int], k: int) -> float:
    """Chunk-level index recall@k against an exact ground truth.

    recall@k = |approx[:k] ∩ exact[:k]| / k. The denominator is always k, so an
    approximate result that returns fewer than k rows (e.g. HNSW truncated by
    ``ef_search < k``) scores below 1.0 honestly rather than being renormalized.
    """
    if k < 1:
        raise ValueError("k must be positive")
    exact_k = set(exact_ids[:k])
    approx_k = set(approx_ids[:k])
    return len(approx_k & exact_k) / k


def recall_at(ranked_ids: list[str], relevant: "set[str] | dict[str, int]", k: int) -> float:
    relevant_ids = set(_as_grades(relevant))
    return float(bool(set(ranked_ids[:k]) & relevant_ids))


def reciprocal_rank(ranked_ids: list[str], relevant: "set[str] | dict[str, int]") -> float:
    relevant_ids = set(_as_grades(relevant))
    for rank, identifier in enumerate(ranked_ids, start=1):
        if identifier in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at(ranked_ids: list[str], relevant: "set[str] | dict[str, int]", k: int) -> float:
    """Graded nDCG@k with gain ``2**grade - 1``.

    A binary set is treated as all-grade-1, which reduces exactly to the previous
    binary formula, so schema 1/2 evidence reproduces unchanged.
    """
    grades = _as_grades(relevant)
    if not grades:
        return 0.0
    dcg = sum(
        (2 ** grades[identifier] - 1) / math.log2(rank + 1)
        for rank, identifier in enumerate(ranked_ids[:k], start=1)
        if identifier in grades
    )
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    ideal = sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return dcg / ideal if ideal else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _load_question_set_v3(payload: dict[str, Any], manifest: CorpusManifest) -> dict[str, Any]:
    """Schema-3 loader: graded relevance, typed questions, counts derived from the file."""
    if payload.get("corpus_arxiv_ids_sha256") != manifest.arxiv_ids_sha256:
        raise ValueError("evaluation set corpus hash does not match manifest")
    corpus_ids = set(manifest.arxiv_ids)
    questions = payload.get("retrieval_questions", [])
    abstention = payload.get("abstention_questions", [])
    if not questions:
        raise ValueError("schema-3 evaluation set has no retrieval questions")

    ids = [item["id"] for item in questions] + [item["id"] for item in abstention]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation question IDs must be unique")

    for item in questions:
        if item.get("split") not in ("development", "heldout"):
            raise ValueError(f"question {item.get('id')} has an invalid split")
        if not item.get("type"):
            raise ValueError(f"question {item.get('id')} is missing a type")
        relevant = item.get("relevant") or {}
        if not isinstance(relevant, dict) or not relevant:
            raise ValueError(f"question {item.get('id')} needs a non-empty graded relevant map")
        for arxiv_id, grade in relevant.items():
            if arxiv_id not in corpus_ids:
                raise ValueError(f"question {item['id']} references {arxiv_id} outside the corpus")
            if int(grade) not in (1, 2):
                raise ValueError(f"question {item['id']} has grade {grade}; expected 1 or 2")
        # Normalize to graded ints and expose the id list for legacy readers.
        item["relevant"] = {arxiv_id: int(grade) for arxiv_id, grade in relevant.items()}
        item["relevant_arxiv_ids"] = sorted(item["relevant"])
        pool = item.get("pool")
        if pool is not None:
            if not set(item["relevant"]).issubset(set(pool)):
                raise ValueError(f"question {item['id']} grades an id outside its pool")

    for item in abstention:
        if item.get("split") not in ("development", "heldout"):
            raise ValueError(f"abstention {item.get('id')} has an invalid split")
        if not item.get("type"):
            raise ValueError(f"abstention {item.get('id')} is missing a type")
    return payload


def load_question_set(path: Path, manifest: CorpusManifest) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in (1, 2, 3):
        raise ValueError("unsupported evaluation schema")
    if payload.get("schema_version") == 3:
        return _load_question_set_v3(payload, manifest)
    if payload.get("development_source"):
        source_path = path.parent / payload["development_source"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        payload["retrieval_questions"] = [
            item
            for item in source["retrieval_questions"]
            if item["split"] == "development"
        ] + payload["heldout_retrieval_questions"]
        payload["abstention_questions"] = [
            item
            for item in source["abstention_questions"]
            if item["split"] == "development"
        ] + payload["heldout_abstention_questions"]
    if payload.get("corpus_arxiv_ids_sha256") != manifest.arxiv_ids_sha256:
        raise ValueError("evaluation set corpus hash does not match manifest")
    questions = payload.get("retrieval_questions", [])
    if len(questions) != 30:
        raise ValueError(f"expected 30 retrieval questions, found {len(questions)}")
    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation question IDs must be unique")
    corpus_ids = set(manifest.arxiv_ids)
    unknown = sorted(
        {
            relevant
            for item in questions
            for relevant in item["relevant_arxiv_ids"]
            if relevant not in corpus_ids
        }
    )
    if unknown:
        raise ValueError(f"evaluation references IDs outside the corpus: {unknown}")
    if sum(item["split"] == "development" for item in questions) != 20:
        raise ValueError("retrieval evaluation must contain 20 development questions")
    if sum(item["split"] == "heldout" for item in questions) != 10:
        raise ValueError("retrieval evaluation must contain 10 held-out questions")
    abstention = payload.get("abstention_questions", [])
    if sum(item["split"] == "development" for item in abstention) != 10:
        raise ValueError("abstention evaluation must contain 10 development negatives")
    if sum(item["split"] == "heldout" for item in abstention) != 5:
        raise ValueError("abstention evaluation must contain 5 held-out negatives")
    return payload


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "questions": len(rows),
        "recall_at_5": statistics.fmean(row["recall_at_5"] for row in rows),
        "mrr": statistics.fmean(row["reciprocal_rank"] for row in rows),
        "ndcg_at_10": statistics.fmean(row["ndcg_at_10"] for row in rows),
        "latency_ms_p50": _percentile([row["latency_ms"] for row in rows], 0.50),
        "latency_ms_p95": _percentile([row["latency_ms"] for row in rows], 0.95),
    }


def _aggregate_typed(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Schema-3 aggregate: the base metrics plus graded Recall@10."""
    aggregate = _aggregate(rows)
    aggregate["recall_at_10"] = statistics.fmean(row["recall_at_10"] for row in rows)
    return aggregate


def _aggregate_by_type(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Return ``{"all": ..., "<type>": ...}`` for a split/mode slice of rows."""
    result: dict[str, dict[str, float | int]] = {"all": _aggregate_typed(rows)}
    for question_type in sorted({row["type"] for row in rows}):
        typed = [row for row in rows if row["type"] == question_type]
        result[question_type] = _aggregate_typed(typed)
    return result


def _choose_threshold(
    positive_scores: list[float], negative_scores: list[float]
) -> dict[str, float]:
    values = sorted(set(positive_scores + negative_scores))
    candidates = [0.0]
    candidates.extend(values)
    candidates.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    candidates.append((values[-1] + 1.0) / 2)
    best: dict[str, float] | None = None
    for threshold in sorted(set(candidates)):
        positive_accept = statistics.fmean(
            float(score >= threshold) for score in positive_scores
        )
        negative_abstain = statistics.fmean(
            float(score < threshold) for score in negative_scores
        )
        balanced = (positive_accept + negative_abstain) / 2
        candidate = {
            "threshold": threshold,
            "development_positive_accept_rate": positive_accept,
            "development_negative_abstain_rate": negative_abstain,
            "development_balanced_accuracy": balanced,
        }
        if best is None or (
            balanced,
            positive_accept,
            -threshold,
        ) > (
            best["development_balanced_accuracy"],
            best["development_positive_accept_rate"],
            -best["threshold"],
        ):
            best = candidate
    assert best is not None
    return best


def evaluate(
    session: Session,
    *,
    question_set: dict[str, Any],
    manifest: CorpusManifest,
    encoder: Encoder,
    output_path: Path,
    modes: "tuple[SearchMode, ...] | None" = None,
    reranker: object | None = None,
    rerank_pool: int | None = None,
) -> dict[str, Any]:
    schema = int(question_set["schema_version"])
    graded_mode = schema >= 3
    modes = tuple(modes) if modes is not None else _modes_for_schema(schema)
    rerank_modes = {"hybrid_rerank", "vector_rerank"}
    has_rerank = any(mode in rerank_modes for mode in modes)

    def _retrieve(query: str, mode: SearchMode, limit: int):
        kwargs: dict[str, Any] = {
            "mode": mode,
            "limit": limit,
            "encoder": encoder if mode != "keyword" else None,
        }
        if mode in rerank_modes:
            kwargs["reranker"] = reranker
            kwargs["rerank_pool"] = rerank_pool
        return retrieve(session, query, **kwargs)

    # Discard one warm-up per path before collecting latency.
    warmup = question_set["retrieval_questions"][0]["query"]
    for mode in modes:
        _retrieve(warmup, mode, 10)

    # Topical questions are judged only within a frozen pool; a rerank mode over a
    # deeper pool can surface first-stage candidates that were never judged. Those
    # are scored grade 0 but recorded per row so the understatement is auditable.
    pools = {
        q["id"]: set(q["pool"])
        for q in question_set["retrieval_questions"]
        if q.get("pool")
    }

    rows: list[dict[str, Any]] = []
    hybrid_scores: dict[str, float] = {}
    for item in question_set["retrieval_questions"]:
        graded = item.get("relevant") or {rid: 1 for rid in item["relevant_arxiv_ids"]}
        for mode in modes:
            started = time.perf_counter()
            hits = _retrieve(item["query"], mode, 10)
            latency_ms = (time.perf_counter() - started) * 1000
            ranked = [str(hit["arxiv_id"]) for hit in hits]
            row = {
                "question_id": item["id"],
                "split": item["split"],
                "mode": mode,
                "relevant_arxiv_ids": item["relevant_arxiv_ids"],
                "ranked_arxiv_ids": ranked,
                "recall_at_5": recall_at(ranked, graded, 5),
                "reciprocal_rank": reciprocal_rank(ranked, graded),
                "ndcg_at_10": ndcg_at(ranked, graded, 10),
                "latency_ms": latency_ms,
                "top_score": float(hits[0]["score"]) if hits else 0.0,
            }
            if graded_mode:
                row["type"] = item["type"]
                row["relevant"] = graded
                row["recall_at_10"] = recall_at(ranked, graded, 10)
                if mode in rerank_modes:
                    row["rerank_pool_size"] = (
                        int(hits[0]["rerank_pool_size"]) if hits else 0
                    )
                    pool = pools.get(item["id"])
                    if pool is not None:
                        row["unjudged_ranked_ids"] = sorted(set(ranked) - pool)
            rows.append(row)
            if mode == "hybrid":
                hybrid_scores[item["id"]] = row["top_score"]

    # The abstention gate is the hybrid RRF top score; compute it independently
    # when hybrid is not among the evaluated modes (e.g. a rerank-only study run)
    # so the frozen threshold is still selected on the same signal.
    if "hybrid" not in modes:
        for item in question_set["retrieval_questions"]:
            hits = retrieve(session, item["query"], mode="hybrid", limit=10, encoder=encoder)
            hybrid_scores[item["id"]] = float(hits[0]["score"]) if hits else 0.0

    negative_scores: dict[str, float] = {}
    for item in question_set["abstention_questions"]:
        hits = retrieve(
            session, item["query"], mode="hybrid", limit=5, encoder=encoder
        )
        negative_scores[item["id"]] = float(hits[0]["score"]) if hits else 0.0

    development_positive = [
        hybrid_scores[item["id"]]
        for item in question_set["retrieval_questions"]
        if item["split"] == "development"
    ]
    development_negative = [
        negative_scores[item["id"]]
        for item in question_set["abstention_questions"]
        if item["split"] == "development"
    ]
    threshold = _choose_threshold(development_positive, development_negative)
    frozen = threshold["threshold"]
    heldout_positive = [
        hybrid_scores[item["id"]]
        for item in question_set["retrieval_questions"]
        if item["split"] == "heldout"
    ]
    heldout_negative = [
        negative_scores[item["id"]]
        for item in question_set["abstention_questions"]
        if item["split"] == "heldout"
    ]
    heldout_positive_accept = statistics.fmean(
        float(score >= frozen) for score in heldout_positive
    )
    heldout_negative_abstain = statistics.fmean(
        float(score < frozen) for score in heldout_negative
    )

    aggregates: dict[str, Any] = {}
    for split in ("development", "heldout"):
        aggregates[split] = {}
        for mode in modes:
            slice_rows = [
                row for row in rows if row["split"] == split and row["mode"] == mode
            ]
            aggregates[split][mode] = (
                _aggregate_by_type(slice_rows) if graded_mode else _aggregate(slice_rows)
            )

    if graded_mode:
        protocol = {
            "schema_version": schema,
            "modes": list(modes),
            "development_retrieval_questions": sum(
                item["split"] == "development"
                for item in question_set["retrieval_questions"]
            ),
            "heldout_retrieval_questions": sum(
                item["split"] == "heldout"
                for item in question_set["retrieval_questions"]
            ),
            "sampling_seed": question_set.get("sampling_seed"),
            "rrf_k": 60,
            "retrieval_limit": 10,
            "latency_warmups_discarded_per_mode": 1,
            "threshold_selected_on": "development positives and development negatives only",
            "threshold_tie_break": "balanced accuracy, then positive accept rate, then lower threshold",
            "topical_grade_provenance": question_set.get("topical_grade_provenance"),
        }
        if has_rerank:
            reranker_model = getattr(reranker, "model_name", None) or (
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            if reranker_model.endswith("-fallback"):
                raise ValueError(
                    "rerank study ran on the lexical fallback, not a cross-encoder (A6)"
                )
            protocol["reranker_model"] = reranker_model
            protocol["rerank_pool"] = (
                rerank_pool if rerank_pool is not None else max(10 * 2, 20)
            )
            max_length = getattr(reranker, "max_length", None)
            if max_length is not None:
                protocol["reranker_max_length"] = int(max_length)
        claim_boundary = (
            "Graded relevance over paraphrase, lexical, and pooled-topical queries "
            "plus out-of-domain and near-miss negatives. Queries were LLM-authored "
            "(Claude); topical relevance grades were LLM-drafted and re-graded blind "
            "by a human before freeze (see topical_grade_disagreement in protocol). "
            "Held-out questions were authored from a fresh sample disjoint from the "
            "development sample and were not dry-run before freeze. Paraphrase queries "
            "are relatively long, abstract-style rewrites and are therefore easier for "
            "dense retrieval than terse user queries. This is not user-traffic relevance."
        )
    else:
        protocol = {
            "known_item_questions": 30,
            "development_questions": 20,
            "heldout_questions": 10,
            "rrf_k": 60,
            "retrieval_limit": 10,
            "latency_warmups_discarded_per_mode": 1,
            "threshold_selected_on": "development positives and development out-of-domain negatives only",
            "threshold_tie_break": "balanced accuracy, then positive accept rate, then lower threshold",
        }
        claim_boundary = "Known-item retrieval over title-derived questions; it is not exhaustive topical relevance or user-traffic evaluation."

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": question_set["frozen_at_utc"],
        "evaluation_schema_version": question_set["schema_version"],
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "embedding_model": encoder.model_name,
        "protocol": protocol,
        "aggregates": aggregates,
        "abstention": {
            **threshold,
            "heldout_positive_accept_rate": heldout_positive_accept,
            "heldout_negative_abstain_rate": heldout_negative_abstain,
            "heldout_balanced_accuracy": (
                heldout_positive_accept + heldout_negative_abstain
            )
            / 2,
            "development_positive_scores": development_positive,
            "development_negative_scores": development_negative,
            "heldout_positive_scores": heldout_positive,
            "heldout_negative_scores": heldout_negative,
        },
        "per_question": rows,
        "claim_boundary": claim_boundary,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _rank_of_relevant(retrieved_ids: list[str], relevant: set[str]) -> int | None:
    for rank, identifier in enumerate(retrieved_ids, start=1):
        if identifier in relevant:
            return rank
    return None


def evaluate_rag(
    session: Session,
    *,
    question_set: dict[str, Any],
    encoder: Encoder,
    generator: Generator,
    threshold: float,
    output_path: Path,
    gate_signal_name: str = "rrf_top",
    verify_entailment: bool = False,
    min_faithfulness: float = 0.80,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    positives = [
        item
        for item in question_set["retrieval_questions"]
        if item["split"] == "heldout"
    ]
    negatives = [
        item
        for item in question_set["abstention_questions"]
        if item["split"] == "heldout"
    ]
    for expected_answerable, items in ((True, positives), (False, negatives)):
        for item in items:
            relevant = set(item.get("relevant") or item.get("relevant_arxiv_ids") or [])
            started = time.perf_counter()
            try:
                result = answer_question(
                    session,
                    item["query"],
                    encoder=encoder,
                    generator=generator,
                    threshold=threshold,
                    top_k=5,
                    gate_signal_name=gate_signal_name,
                    verify_entailment=verify_entailment,
                    min_faithfulness=min_faithfulness,
                )
                rank = (
                    _rank_of_relevant(list(result.retrieved_arxiv_ids), relevant)
                    if expected_answerable
                    else None
                )
                records.append(
                    {
                        "question_id": item["id"],
                        "type": item.get("type"),
                        "expected_answerable": expected_answerable,
                        "abstained": result.abstained,
                        "top_score": result.top_score,
                        "gate_score": result.gate_score,
                        "answer": result.answer,
                        # No emitted answer (threshold or entailment refusal) => no
                        # emitted citations, so the record stays parse-consistent.
                        "citations": list(result.citations) if result.answer else [],
                        "retrieved_arxiv_ids": list(result.retrieved_arxiv_ids),
                        "retrieved_rank_of_relevant": rank,
                        "entailment_verified": result.entailment_verified,
                        "faithfulness_score": result.faithfulness_score,
                        "entailment_refused": bool(result.abstained and result.entailment_verified),
                        "citations_all_retrieved": bool(result.answer and result.citations)
                        and set(result.citations).issubset(result.retrieved_arxiv_ids),
                        "error": None,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                    }
                )
            except Exception as error:  # preserve provider and enforcement failures as evidence
                records.append(
                    {
                        "question_id": item["id"],
                        "type": item.get("type"),
                        "expected_answerable": expected_answerable,
                        "abstained": False,
                        "top_score": None,
                        "gate_score": None,
                        "answer": None,
                        "citations": [],
                        "retrieved_arxiv_ids": [],
                        "retrieved_rank_of_relevant": None,
                        "entailment_verified": False,
                        "faithfulness_score": None,
                        "entailment_refused": False,
                        "citations_all_retrieved": False,
                        "error": f"{type(error).__name__}: {error}",
                        "latency_ms": (time.perf_counter() - started) * 1000,
                    }
                )
    positive_records = [row for row in records if row["expected_answerable"]]
    negative_records = [row for row in records if not row["expected_answerable"]]
    ood = [row for row in negative_records if row.get("type") == "negative_ood"]
    near = [row for row in negative_records if row.get("type") == "negative_near"]
    generated = [
        row for row in positive_records if not row["abstained"] and row["answer"]
    ]
    report = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": question_set["frozen_at_utc"],
        "generator_model": generator.model_name,
        "embedding_model": encoder.model_name,
        "gate_signal": gate_signal_name,
        "verify_entailment": verify_entailment,
        "frozen_abstain_threshold": threshold,
        "heldout_answerable_questions": len(positive_records),
        "heldout_out_of_domain_questions": len(negative_records),
        "answerable_answer_rate": statistics.fmean(
            float(bool(row["answer"])) for row in positive_records
        ),
        "answerable_abstain_rate": statistics.fmean(
            float(row["abstained"]) for row in positive_records
        ),
        "answerable_refused_at_rank_1or2": sum(
            row["abstained"] and row["retrieved_rank_of_relevant"] in (1, 2)
            for row in positive_records
        ),
        "out_of_domain_abstain_rate": statistics.fmean(
            float(row["abstained"]) for row in negative_records
        ),
        "negative_ood_abstain_rate": statistics.fmean(float(row["abstained"]) for row in ood) if ood else 0.0,
        "negative_near_abstain_rate": statistics.fmean(float(row["abstained"]) for row in near) if near else 0.0,
        "entailment_refusals": sum(bool(row["entailment_refused"]) for row in records),
        "provider_or_enforcement_errors": sum(bool(row["error"]) for row in records),
        "citation_grounding_rate_among_answers": (
            statistics.fmean(float(row["citations_all_retrieved"]) for row in generated)
            if generated
            else 0.0
        ),
        "records": records,
        "claim_boundary": "Citation grounding means every emitted versioned ID was in the retrieved set; it does not independently verify every natural-language statement.",
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
