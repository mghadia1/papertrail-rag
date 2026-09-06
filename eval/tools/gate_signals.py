"""Phase 1b D1: collect abstention-gate signals for every v3 question.

Answerable = the 78 retrieval questions; not-answerable = the 27 abstention
questions. For each, record several candidate confidence signals so D2 can pick
the one that best separates answerable from not-answerable. Both splits are
collected here; SELECTION happens in D2 on development only.

Signals (higher = more confident it is answerable):
  rrf_top        current gate: hybrid RRF top score
  cos_top        top vector score (1 - cosine distance)
  cos_margin     cos_top minus the 2nd vector score
  cos_mean_top3  mean of the top-3 vector scores
  kw_top         top keyword score (0.0 if keyword returns nothing)
  ce_top         hybrid_rerank top cross-encoder rerank_score (a logit)
  ce_margin      ce_top minus the 2nd rerank_score
  ce_sigmoid_top sigmoid(ce_top)

For answerable questions, ``retrieved_rank_of_relevant`` is the lowest hybrid
rank (in the top 50) at which any graded-relevant paper appears, or null — so D3
can separate "refused although retrieved at rank 1" from "not retrieved".

Writes docs/evidence/phase-8-gate-signals.json. Asserts the reranker that ran is
the real cross-encoder, not a fallback (A6).
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

from papertrail.database import session_scope
from papertrail.embedding import get_encoder
from papertrail.evaluation import load_question_set
from papertrail.manifest import CorpusManifest
from papertrail.repository import keyword_search, vector_search
from papertrail.retrieval import retrieve

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "evidence" / "corpus-manifest-1000.json"
QUESTIONS = ROOT / "eval" / "questions-v3.json"
OUT = ROOT / "docs" / "evidence" / "phase-8-gate-signals.json"


def _rank_of_relevant(session, query, encoder, relevant: set[str]) -> int | None:
    hits = retrieve(session, query, mode="hybrid", limit=50, encoder=encoder)
    for rank, hit in enumerate(hits, start=1):
        if str(hit["arxiv_id"]) in relevant:
            return rank
    return None


def _signals(session, query, encoder) -> tuple[dict, str]:
    emb = encoder.encode([query], batch_size=1)[0]
    hybrid = retrieve(session, query, mode="hybrid", limit=5, encoder=encoder)
    vhits = vector_search(session, emb, limit=50)
    khits = keyword_search(session, query, limit=50)
    rr = retrieve(session, query, mode="hybrid_rerank", limit=5, encoder=encoder)

    cos_scores = [float(h["score"]) for h in vhits]
    ce_scores = [float(h["rerank_score"]) for h in rr]
    reranker_model = str(rr[0]["reranker"]) if rr else ""
    ce_top = ce_scores[0] if ce_scores else 0.0
    row = {
        "rrf_top": float(hybrid[0]["score"]) if hybrid else 0.0,
        "cos_top": cos_scores[0] if cos_scores else 0.0,
        "cos_margin": (cos_scores[0] - cos_scores[1]) if len(cos_scores) >= 2 else 0.0,
        "cos_mean_top3": statistics.fmean(cos_scores[:3]) if cos_scores else 0.0,
        "kw_top": float(khits[0]["score"]) if khits else 0.0,
        "ce_top": ce_top,
        "ce_margin": (ce_scores[0] - ce_scores[1]) if len(ce_scores) >= 2 else 0.0,
        "ce_sigmoid_top": 1.0 / (1.0 + math.exp(-ce_top)),
    }
    return row, reranker_model


SIGNAL_NAMES = ["rrf_top", "cos_top", "cos_margin", "cos_mean_top3", "kw_top",
                "ce_top", "ce_margin", "ce_sigmoid_top"]


def main() -> int:
    manifest = CorpusManifest.read(MANIFEST)
    question_set = load_question_set(QUESTIONS, manifest)
    encoder = get_encoder()
    rows: list[dict] = []
    reranker_models: set[str] = set()
    with session_scope() as session:
        for q in question_set["retrieval_questions"]:
            sig, model = _signals(session, q["query"], encoder)
            reranker_models.add(model)
            rows.append({
                "question_id": q["id"], "split": q["split"], "type": q["type"],
                "is_answerable": True,
                "retrieved_rank_of_relevant": _rank_of_relevant(
                    session, q["query"], encoder, set(q["relevant"])),
                **sig,
            })
        for q in question_set["abstention_questions"]:
            sig, model = _signals(session, q["query"], encoder)
            reranker_models.add(model)
            rows.append({
                "question_id": q["id"], "split": q["split"], "type": q["type"],
                "is_answerable": False, "retrieved_rank_of_relevant": None, **sig,
            })

    if len(reranker_models) != 1:
        raise SystemExit(f"expected one reranker model, saw {reranker_models}")
    reranker_model = reranker_models.pop()
    if reranker_model.endswith("-fallback") or not reranker_model:  # A6
        raise SystemExit(f"reranker fell back to a non-cross-encoder: {reranker_model!r}")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": question_set["frozen_at_utc"],
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "embedding_model": encoder.model_name,
        "reranker_model": reranker_model,
        "protocol": {
            "signals": SIGNAL_NAMES,
            "answerable_questions": sum(r["is_answerable"] for r in rows),
            "negative_questions": sum(not r["is_answerable"] for r in rows),
            "note": "collection only; signal selection happens on development in score_gate.py",
        },
        "rows": rows,
        "claim_boundary": "Gate signals on the frozen v3 questions; higher = more confident answerable.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "rows": len(rows),
                      "reranker_model": reranker_model,
                      "answerable": report["protocol"]["answerable_questions"],
                      "negatives": report["protocol"]["negative_questions"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
