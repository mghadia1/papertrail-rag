"""Postgres keyword-variant ablation (brief Part F, F2). Development split only.

Runs the SQL keyword path directly (``keyword_search`` -> ``distinct_papers``) so
every variant is measured by exactly the same harness as
``eval/tools/bm25_ablation.py``. That matters: the F3 comparison table puts BM25
and the Postgres variants in the same rows, and running them through different
harnesses would confound the comparison with candidate-depth and collapse
differences.

Development only, by design. Held-out is reserved for one final report on the
best development configuration (brief F3 / rule A1), and the verifier rejects a
held-out row in these files.

Variants:
  or                baseline reproduction of the frozen v3 keyword mode
  or-depth200       same, but 200 candidate chunks — isolates the candidate-depth
                    difference against the BM25 run, which used 200
  weighted-n<N>     field-weighted column (title A, body B) ranked with
                    ts_rank_cd({0.1,0.2,0.4,1.0}, ..., N)

    python eval/tools/keyword_variants.py [variant ...]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from papertrail.database import session_scope
from papertrail.evaluation import (
    _aggregate_by_type,
    load_question_set,
    ndcg_at,
    recall_at,
    reciprocal_rank,
)
from papertrail.manifest import CorpusManifest
from papertrail.repository import keyword_search
from papertrail.retrieval import distinct_papers

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "evidence" / "corpus-manifest-1000.json"
QUESTIONS = ROOT / "eval" / "questions-v3.json"
EVIDENCE = ROOT / "docs" / "evidence"

LIMIT = 10
# `retrieve()` uses min(200, max(50, limit * 10)) = 100 candidate chunks at limit 10.
DEFAULT_CANDIDATES = 100

# name -> (candidate_chunks, weighted, normalization, strategy)
VARIANTS: dict[str, tuple[int, bool, int, str]] = {
    "or": (DEFAULT_CANDIDATES, False, 0, "or"),
    "or-depth200": (200, False, 0, "or"),
    "weighted-n0": (DEFAULT_CANDIDATES, True, 0, "or"),
    "weighted-n1": (DEFAULT_CANDIDATES, True, 1, "or"),
    "weighted-n2": (DEFAULT_CANDIDATES, True, 2, "or"),
    "cascade": (DEFAULT_CANDIDATES, False, 0, "cascade"),
    "cascade-weighted-n0": (DEFAULT_CANDIDATES, True, 0, "cascade"),
    "cascade-phrase": (DEFAULT_CANDIDATES, False, 0, "cascade_phrase"),
}


def run_variant(session, question_set, manifest, name: str, split: str = "development") -> dict:
    candidates, weighted, normalization, strategy = VARIANTS[name]
    pools = {
        q["id"]: set(q["pool"])
        for q in question_set["retrieval_questions"]
        if q.get("pool")
    }
    items = [
        q for q in question_set["retrieval_questions"] if q["split"] == split
    ]

    # One discarded warm-up so the first query's plan/cache cost is not measured.
    keyword_search(session, items[0]["query"], limit=candidates,
                   weighted=weighted, normalization=normalization, strategy=strategy)

    rows: list[dict] = []
    for item in items:
        graded = {str(k): int(v) for k, v in item["relevant"].items()}
        started = time.perf_counter()
        hits = keyword_search(
            session, item["query"], limit=candidates,
            weighted=weighted, normalization=normalization, strategy=strategy,
        )
        selected = distinct_papers(hits, limit=LIMIT)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = [str(h["arxiv_id"]) for h in selected]
        row = {
            "question_id": item["id"],
            "split": item["split"],
            "mode": f"keyword-{name}",
            "type": item["type"],
            "relevant_arxiv_ids": item["relevant_arxiv_ids"],
            "relevant": graded,
            "ranked_arxiv_ids": ranked,
            "recall_at_5": recall_at(ranked, graded, 5),
            "recall_at_10": recall_at(ranked, graded, 10),
            "reciprocal_rank": reciprocal_rank(ranked, graded),
            "ndcg_at_10": ndcg_at(ranked, graded, 10),
            "top_score": float(selected[0]["score"]) if selected else 0.0,
            "matched_chunks": len(hits),
            "latency_ms": latency_ms,
        }
        pool = pools.get(item["id"])
        if pool is not None:
            row["unjudged_ranked_ids"] = sorted(set(ranked) - pool)
        rows.append(row)

    rank_expression = (
        f"ts_rank_cd('{{0.1,0.2,0.4,1.0}}', search_vector_weighted, query, {normalization})"
        if weighted
        else ("ts_rank_cd(search_vector, query)" if not normalization
              else f"ts_rank_cd(search_vector, query, {normalization})")
    )
    report = {
        "schema_version": 1,
        "kind": "keyword_variant",
        "variant": name,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": question_set["frozen_at_utc"],
        "evaluation_schema_version": question_set["schema_version"],
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "protocol": {
            "keyword_strategy": strategy,
            "query_construction": (
                "websearch_to_tsquery('english', ' OR '.join(terms))" if strategy == "or"
                else ("phraseto_tsquery spans first, then " if strategy == "cascade_phrase" else "")
                     + "to_tsquery('english', \" & \".join(quoted terms)) then OR-fill to limit"
            ),
            "rank_expression": rank_expression,
            "search_vector_column": "search_vector_weighted" if weighted else "search_vector",
            "ts_rank_cd_normalization": normalization,
            "candidate_chunks": candidates,
            "retrieval_limit": LIMIT,
            "splits_evaluated": [split],
            "split_note": (
                "Held-out is reserved for one final report on the best development "
                "configuration (brief F3, rule A1)."
                if split == "development"
                else "Single final held-out confirmation of the best development "
                     "configuration (brief F3); no tuning was done on this split."
            ),
            "tokenizer": "[A-Za-z0-9]+(?:-[A-Za-z0-9]+)* lower-cased; terms <3 chars dropped",
            "stemming": "Postgres english configuration (stems)",
            "latency_warmups_discarded": 1,
        },
        "aggregates": {split: {f"keyword-{name}": _aggregate_by_type(rows)}},
        "per_question": rows,
        "claim_boundary": (
            "Postgres keyword retrieval on the development split only. Latencies are "
            "within-file only (A10). Topical relevance is judged within a frozen pool "
            "built from four other retrievers, so any paper this variant surfaces that "
            "nobody judged is scored grade 0 and listed in unjudged_ranked_ids."
        ),
    }
    suffix = "" if split == "development" else f"-{split}"
    out = EVIDENCE / f"phase-8-keyword-{name}{suffix}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"variant": name, "output": str(out.relative_to(ROOT)),
            "aggregates": report["aggregates"][split][f"keyword-{name}"]}


def main() -> int:
    argv = sys.argv[1:]
    split = "development"
    if "--heldout" in argv:
        argv.remove("--heldout")
        split = "heldout"
    names = argv or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {unknown}; choose from {list(VARIANTS)}")
    manifest = CorpusManifest.read(MANIFEST)
    question_set = load_question_set(QUESTIONS, manifest)
    results = []
    with session_scope() as session:
        for name in names:
            results.append(run_variant(session, question_set, manifest, name, split))
    for r in results:
        a = r["aggregates"]
        print(f"{r['variant']:14s} all={a['all']['ndcg_at_10']:.3f} "
              f"lex={a['lexical']['ndcg_at_10']:.3f} "
              f"par={a['paraphrase']['ndcg_at_10']:.3f} "
              f"top={a['topical']['ndcg_at_10']:.3f} -> {r['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
