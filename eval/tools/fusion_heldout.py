"""Held-out confirmation for Part G, corrected re-run (review F3).

Supersedes ``docs/evidence/phase-8-fusion-heldout.json``. That first run executed
both configurations inside one ``session_scope``; because ``SET LOCAL`` lasts for
the whole transaction, the baseline inherited ``hnsw.ef_search = 200`` from the
chosen configuration and reported 50 vector candidates instead of production's 40.
The old file is kept (A3) and is superseded by this one.

Two changes make that impossible here:

* every configuration runs in **its own** ``session_scope``, and
* ``vector_search`` now resets every GUC it touches on every call (A12b).

Three configurations, each run once on held-out:

  production-as-is     k=60, pool 50, ef_search unset. This is what the live
                       system does, and HNSW caps it at 40 candidates.
  production-ef-fixed  k=60, pool 50, ef_search=50 — isolates the cap alone.
  chosen               k=10, pool 200, ef_search=200 — the sweep's development
                       winner, read from the sweep file rather than hard-coded.

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
OUT = ROOT / "docs" / "evidence" / "phase-8-fusion-heldout-v2.json"
SUPERSEDES = "docs/evidence/phase-8-fusion-heldout.json"

LIMIT = 10

PRODUCTION_AS_IS = {
    "fusion": "rrf", "k": 60, "w_vec": 1.0, "w_kw": 1.0,
    "candidate_limit": 50, "ef_search": None, "keyword_strategy": "or",
    # HNSW returns at most hnsw.ef_search rows and the server default is 40, so
    # this configuration cannot see more than 40 vector candidates however many
    # candidate_limit asks for. The verifier asserts this per row.
    "expected_vector_candidates": 40,
}
PRODUCTION_EF_FIXED = {
    "fusion": "rrf", "k": 60, "w_vec": 1.0, "w_kw": 1.0,
    "candidate_limit": 50, "ef_search": 50, "keyword_strategy": "or",
    "expected_vector_candidates": 50,
}


def run(encoder, items, pools, config, label):
    """One configuration, in its own transaction (review F3)."""
    rows = []
    ef = config["ef_search"]
    ef = int(ef) if ef is not None else None
    cl = int(config["candidate_limit"])
    strategy = config["keyword_strategy"]
    embeddings = {q["id"]: encoder.encode([q["query"]], batch_size=1)[0] for q in items}

    with session_scope() as session:
        require_vector_search_ready(
            session, model_name=encoder.model_name, dimensions=encoder.dimensions
        )
        vector_search(session, embeddings[items[0]["id"]], limit=cl, ef_search=ef)
        keyword_search(session, items[0]["query"], limit=cl, strategy=strategy)

        for item in items:
            started = time.perf_counter()
            vec = vector_search(session, embeddings[item["id"]], limit=cl, ef_search=ef)
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
    chosen["expected_vector_candidates"] = int(chosen["candidate_limit"])

    encoder = get_encoder()
    items = [q for q in question_set["retrieval_questions"] if q["split"] == "heldout"]
    pools = {q["id"]: set(q["pool"]) for q in question_set["retrieval_questions"] if q.get("pool")}

    configs = {
        "production-as-is": PRODUCTION_AS_IS,
        "production-ef-fixed": PRODUCTION_EF_FIXED,
        "chosen": chosen,
    }
    rows: list[dict] = []
    for label, config in configs.items():
        rows += run(encoder, items, pools, config, label)

    aggregates = {
        "heldout": {
            label: _aggregate_by_type([r for r in rows if r["config_id"] == label])
            for label in configs
        }
    }
    report = {
        "schema_version": 2,
        "kind": "fusion_heldout",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": question_set["frozen_at_utc"],
        "evaluation_schema_version": question_set["schema_version"],
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "embedding_model": encoder.model_name,
        "supersedes": SUPERSEDES,
        "supersedes_reason": (
            "The first held-out run executed both configurations in one transaction, "
            "so the baseline inherited hnsw.ef_search=200 from the chosen "
            "configuration (SET LOCAL is transaction-scoped) and saw 50 vector "
            "candidates instead of production's 40. Every configuration here runs in "
            "its own session and vector_search now resets every GUC it touches."
        ),
        "chosen_config_id": chosen_id,
        "chosen_config": {k: v for k, v in chosen.items() if k != "expected_vector_candidates"},
        "configs": configs,
        "sweep_evidence": "docs/evidence/phase-8-fusion-sweep-dev.json",
        "protocol": {
            "splits_evaluated": ["heldout"],
            "retrieval_limit": LIMIT,
            "runs": "one run per configuration, each in its own transaction, no tuning on this split (A1)",
            "selection_rule": sweep["protocol"]["selection_rule"],
            "ef_search_note": (
                "production-as-is leaves ef_search unset, which is what the live "
                "system does; HNSW then returns at most the server default of 40 rows "
                "regardless of candidate_limit. production-ef-fixed differs only by "
                "setting ef_search=50, isolating the cap."
            ),
            "latency_warmups_discarded_per_config": 1,
        },
        "aggregates": aggregates,
        "per_question": rows,
        "claim_boundary": (
            "Three configurations, one run each, on the held-out split (26 questions: "
            "12 paraphrase, 8 lexical, 6 topical). Latencies are within-file only "
            "(A10). Topical relevance is judged within a frozen pool built from four "
            "retrievers, so a paper surfaced that nobody judged is scored grade 0 and "
            "listed in unjudged_ranked_ids."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "chosen": chosen_id,
                      "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
