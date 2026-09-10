"""Fusion ablation sweep (brief Part G, G2). Development split only.

Design note that matters for reading the numbers: for each question the two
candidate lists are fetched **once** per (candidate_limit, keyword_strategy) and
every fusion configuration is then applied to those same cached lists in memory.
That is deliberate — it means each config sees byte-identical candidates, so a
difference between cells is caused by the fusion alone and never by a re-query.

The consequence is that **per-config latency is not measured and is not
reported**. Retrieval cost depends on candidate_limit and keyword_strategy (two
SQL queries plus one embedding), not on k, the weights, or alpha; the fusion step
itself is an in-memory pass over at most 400 rows. Candidate-fetch latency is
recorded per (candidate_limit, keyword_strategy) instead.

`ef_search` is set to max(candidate_limit, 40) on every vector query, because
HNSW returns at most ef_search rows and the server default of 40 would otherwise
silently truncate a 50- or 200-row candidate list (Phase 1 finding, rule A13).

    python eval/tools/fusion_sweep.py
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
    select_fusion_config,
    load_question_set,
    ndcg_at,
    recall_at,
    reciprocal_rank,
)
from papertrail.manifest import CorpusManifest
from papertrail.repository import keyword_search, require_vector_search_ready, vector_search
from papertrail.retrieval import convex_fusion, distinct_papers, reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "evidence" / "corpus-manifest-1000.json"
QUESTIONS = ROOT / "eval" / "questions-v3.json"
OUT = ROOT / "docs" / "evidence" / "phase-8-fusion-sweep-dev.json"

LIMIT = 10
CANDIDATE_LIMITS = (50, 200)
KEYWORD_STRATEGIES = ("or", "cascade")
RRF_K = (10, 30, 60, 100)
VECTOR_WEIGHTS = (1, 2, 3)
ALPHAS = (0.5, 0.7, 0.9)


def fusion_configs() -> list[dict]:
    configs: list[dict] = []
    for k in RRF_K:
        for w_vec in VECTOR_WEIGHTS:
            configs.append({"fusion": "rrf", "k": k, "w_vec": float(w_vec), "w_kw": 1.0})
    for alpha in ALPHAS:
        configs.append({"fusion": "convex", "alpha": alpha})
    return configs


def config_id(cfg: dict, candidate_limit: int, strategy: str) -> str:
    core = (
        f"rrf-k{cfg['k']}-wv{int(cfg['w_vec'])}"
        if cfg["fusion"] == "rrf"
        else f"convex-a{cfg['alpha']}"
    )
    return f"{core}-cl{candidate_limit}-{strategy}"


def main() -> int:
    manifest = CorpusManifest.read(MANIFEST)
    question_set = load_question_set(QUESTIONS, manifest)
    encoder = get_encoder()
    items = [q for q in question_set["retrieval_questions"] if q["split"] == "development"]
    pools = {q["id"]: set(q["pool"]) for q in question_set["retrieval_questions"] if q.get("pool")}
    configs = fusion_configs()

    rows: list[dict] = []
    fetch_latency: dict[str, list[float]] = {}

    with session_scope() as session:
        require_vector_search_ready(
            session, model_name=encoder.model_name, dimensions=encoder.dimensions
        )
        embeddings = {q["id"]: encoder.encode([q["query"]], batch_size=1)[0] for q in items}

        for candidate_limit in CANDIDATE_LIMITS:
            ef = max(candidate_limit, 40)
            for strategy in KEYWORD_STRATEGIES:
                key = f"cl{candidate_limit}-{strategy}"
                fetch_latency[key] = []
                # Warm-up, discarded.
                vector_search(session, embeddings[items[0]["id"]], limit=candidate_limit, ef_search=ef)
                keyword_search(session, items[0]["query"], limit=candidate_limit, strategy=strategy)

                cache: dict[str, dict] = {}
                for item in items:
                    started = time.perf_counter()
                    vec = vector_search(
                        session, embeddings[item["id"]], limit=candidate_limit, ef_search=ef
                    )
                    kw = keyword_search(
                        session, item["query"], limit=candidate_limit, strategy=strategy
                    )
                    fetch_latency[key].append((time.perf_counter() - started) * 1000)
                    cache[item["id"]] = {"vector": vec, "keyword": kw}

                for cfg in configs:
                    cid = config_id(cfg, candidate_limit, strategy)
                    for item in items:
                        rankings = cache[item["id"]]
                        if cfg["fusion"] == "convex":
                            fused = convex_fusion(rankings, alpha=cfg["alpha"])
                        else:
                            fused = reciprocal_rank_fusion(
                                rankings,
                                k=cfg["k"],
                                weights={"vector": cfg["w_vec"], "keyword": cfg["w_kw"]},
                            )
                        selected = distinct_papers(fused, limit=LIMIT)
                        ranked = [str(h["arxiv_id"]) for h in selected]
                        graded = {str(a): int(b) for a, b in item["relevant"].items()}
                        row = {
                            "question_id": item["id"],
                            "split": "development",
                            "mode": "hybrid",
                            "config_id": cid,
                            "type": item["type"],
                            "relevant": graded,
                            "ranked_arxiv_ids": ranked,
                            "recall_at_5": recall_at(ranked, graded, 5),
                            "recall_at_10": recall_at(ranked, graded, 10),
                            "reciprocal_rank": reciprocal_rank(ranked, graded),
                            "ndcg_at_10": ndcg_at(ranked, graded, 10),
                            "top_score": float(selected[0]["score"]) if selected else 0.0,
                            "vector_candidates": len(rankings["vector"]),
                            "keyword_candidates": len(rankings["keyword"]),
                        }
                        pool = pools.get(item["id"])
                        if pool is not None:
                            row["unjudged_ranked_ids"] = sorted(set(ranked) - pool)
                        rows.append(row)

    by_config: dict[str, dict] = {}
    for cfg in configs:
        for candidate_limit in CANDIDATE_LIMITS:
            for strategy in KEYWORD_STRATEGIES:
                cid = config_id(cfg, candidate_limit, strategy)
                sel = [r for r in rows if r["config_id"] == cid]
                by_config[cid] = {
                    **cfg,
                    "candidate_limit": candidate_limit,
                    "ef_search": max(candidate_limit, 40),
                    "keyword_strategy": strategy,
                    "aggregates": _aggregate_by_type(sel),
                }

    report = {
        "schema_version": 1,
        "kind": "fusion_sweep",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": question_set["frozen_at_utc"],
        "evaluation_schema_version": question_set["schema_version"],
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "embedding_model": encoder.model_name,
        "protocol": {
            "splits_evaluated": ["development"],
            "retrieval_limit": LIMIT,
            "candidate_limits": list(CANDIDATE_LIMITS),
            "ef_search_rule": "max(candidate_limit, 40) so HNSW never truncates the candidate list",
            "keyword_strategies": list(KEYWORD_STRATEGIES),
            "rrf_k_values": list(RRF_K),
            "vector_weights": list(VECTOR_WEIGHTS),
            "keyword_weight": 1.0,
            "convex_alphas": list(ALPHAS),
            "configurations": len(by_config),
            "candidates_fetched_once_per": "(candidate_limit, keyword_strategy, question)",
            "latency_not_measured_per_config": (
                "Every configuration is fused from one cached candidate fetch, so "
                "per-config latency is not measured and is not reported."
            ),
            "candidate_fetch_latency_ms": {
                key: {
                    "p50": sorted(v)[len(v) // 2],
                    "mean": sum(v) / len(v),
                    "n": len(v),
                }
                for key, v in fetch_latency.items()
            },
            "selection_rule": (
                "Highest development nDCG@10 on 'all'; ties by paraphrase, then "
                "lexical; remaining ties to the simplest (unweighted RRF over "
                "weighted, smaller pool over larger). Pre-registered in lab-notes "
                "before the sweep ran (brief G3)."
            ),
        },
        "by_config": by_config,
        "per_question": rows,
        "claim_boundary": (
            "Development-split sweep. This is a search over configurations, NOT a "
            "reportable result: the best cell here is selected on the same data it "
            "was measured on. Only the held-out confirmation (phase-8-fusion-heldout"
            ".json) is reportable. Topical relevance is judged within a frozen pool "
            "built from four retrievers, so any paper a configuration surfaces that "
            "nobody judged is scored grade 0 and listed in unjudged_ranked_ids."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "configs": len(by_config),
                      "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
