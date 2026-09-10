"""Held-out confirmation of the sweep-selected fusion config (brief Part G, G4).

Runs exactly two configurations on the held-out split, once:

  * the configuration the pre-registered G3 rule selects from the development
    sweep — read from the sweep file, never hard-coded, so the verifier can check
    that this file's configuration really is the dev-best (G5);
  * the v2-style baseline the brief specifies: RRF k=60, equal weights, candidate
    pool 50, ``ef_search`` left at the server default of 40, OR keyword — what the
    system does today.

    python eval/tools/fusion_heldout.py
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from papertrail.database import session_scope
from papertrail.embedding import get_encoder
from papertrail.evaluation import (
    _aggregate_by_type,
    load_question_set,
    ndcg_at,
    recall_at,
    reciprocal_rank,
    select_fusion_config,
)
from papertrail.manifest import CorpusManifest
from papertrail.repository import keyword_search, require_vector_search_ready, vector_search
from papertrail.retrieval import convex_fusion, distinct_papers, reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "evidence" / "corpus-manifest-1000.json"
QUESTIONS = ROOT / "eval" / "questions-v3.json"
SWEEP = ROOT / "docs" / "evidence" / "phase-8-fusion-sweep-dev.json"
OUT = ROOT / "docs" / "evidence" / "phase-8-fusion-heldout.json"

LIMIT = 10
# ef_search is left UNSET so Postgres applies its default of 40 — what the live
# system does. Passing 40 explicitly would trip the ef>=limit guard added in
# Phase 1. Measured, this does NOT truncate at this corpus size: EXPLAIN shows the
# planner runs an exact Seq Scan for LIMIT 50 and LIMIT 200 rather than using the
# HNSW index, so all requested rows come back (per-row vector_candidates confirms
# 50 and 200). The ef cap only binds when the index is actually used.
BASELINE = {
    "fusion": "rrf", "k": 60, "w_vec": 1.0, "w_kw": 1.0,
    "candidate_limit": 50, "ef_search": None, "ef_search_effective": 40,
    "keyword_strategy": "or",
}


def run(session, encoder, items, pools, config, label):
    rows = []
    ef = config["ef_search"]
    ef = int(ef) if ef is not None else None
    cl = int(config["candidate_limit"])
    strategy = config["keyword_strategy"]
    # Warm-up, discarded.
    vector_search(session, encoder.encode([items[0]["query"]], batch_size=1)[0], limit=cl, ef_search=ef)
    keyword_search(session, items[0]["query"], limit=cl, strategy=strategy)

    for item in items:
        started = time.perf_counter()
        vec = vector_search(
            session, encoder.encode([item["query"]], batch_size=1)[0], limit=cl, ef_search=ef
        )
        kw = keyword_search(session, item["query"], limit=cl, strategy=strategy)
        rankings = {"vector": vec, "keyword": kw}
        if config["fusion"] == "convex":
            fused = convex_fusion(rankings, alpha=config["alpha"])
        else:
            fused = reciprocal_rank_fusion(
                rankings, k=int(config["k"]),
                weights={"vector": float(config["w_vec"]), "keyword": float(config["w_kw"])},
            )
        selected = distinct_papers(fused, limit=LIMIT)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = [str(h["arxiv_id"]) for h in selected]
        graded = {str(a): int(b) for a, b in item["relevant"].items()}
        row = {
            "question_id": item["id"], "split": "heldout", "mode": "hybrid",
            "config_id": label, "type": item["type"], "relevant": graded,
            "ranked_arxiv_ids": ranked,
            "recall_at_5": recall_at(ranked, graded, 5),
            "recall_at_10": recall_at(ranked, graded, 10),
            "reciprocal_rank": reciprocal_rank(ranked, graded),
            "ndcg_at_10": ndcg_at(ranked, graded, 10),
            "top_score": float(selected[0]["score"]) if selected else 0.0,
            "vector_candidates": len(vec), "keyword_candidates": len(kw),
            "latency_ms": latency_ms,
        }
        pool = pools.get(item["id"])
        if pool is not None:
            row["unjudged_ranked_ids"] = sorted(set(ranked) - pool)
        rows.append(row)
    return rows


def main() -> int:
    manifest = CorpusManifest.read(MANIFEST)
    question_set = load_question_set(QUESTIONS, manifest)
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    chosen_id = select_fusion_config(sweep["by_config"])
    chosen = {k: v for k, v in sweep["by_config"][chosen_id].items() if k != "aggregates"}

    encoder = get_encoder()
    items = [q for q in question_set["retrieval_questions"] if q["split"] == "heldout"]
    pools = {q["id"]: set(q["pool"]) for q in question_set["retrieval_questions"] if q.get("pool")}

    rows: list[dict] = []
    with session_scope() as session:
        require_vector_search_ready(
            session, model_name=encoder.model_name, dimensions=encoder.dimensions
        )
        rows += run(session, encoder, items, pools, chosen, "chosen")
        rows += run(session, encoder, items, pools, BASELINE, "baseline")

    aggregates = {
        "heldout": {
            label: _aggregate_by_type([r for r in rows if r["config_id"] == label])
            for label in ("chosen", "baseline")
        }
    }
    report = {
        "schema_version": 1,
        "kind": "fusion_heldout",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": question_set["frozen_at_utc"],
        "evaluation_schema_version": question_set["schema_version"],
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "embedding_model": encoder.model_name,
        "chosen_config_id": chosen_id,
        "chosen_config": chosen,
        "baseline_config": BASELINE,
        "sweep_evidence": "docs/evidence/phase-8-fusion-sweep-dev.json",
        "protocol": {
            "splits_evaluated": ["heldout"],
            "retrieval_limit": LIMIT,
            "runs": "one run per configuration, no tuning on this split (A1)",
            "selection_rule": sweep["protocol"]["selection_rule"],
            "baseline_note": (
                "The v2-style baseline leaves ef_search at the Postgres default of "
                "40, which is what the live system runs. Measured here, that does "
                "NOT truncate: EXPLAIN shows an exact Seq Scan at LIMIT 50 and 200 "
                "rather than an HNSW index scan, so every requested candidate is "
                "returned (see per-row vector_candidates). At 2,039 vectors the "
                "Phase 1 ef-truncation concern does not bind, because the planner "
                "does not use the ANN index at these limits."
            ),
            "latency_warmups_discarded_per_config": 1,
        },
        "aggregates": aggregates,
        "per_question": rows,
        "claim_boundary": (
            "Two configurations, one run each, on the held-out split. Latencies are "
            "within-file only (A10). Topical relevance is judged within a frozen "
            "pool built from four retrievers, so a paper surfaced that nobody judged "
            "is scored grade 0 and listed in unjudged_ranked_ids."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "chosen": chosen_id,
                      "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
