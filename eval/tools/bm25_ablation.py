"""Offline BM25 ablation against the Postgres FTS keyword mode (brief Part F, F1).

Read-only: this reads chunks from the database and writes one evidence file. It
makes no DB writes and changes no default.

Protocol notes that matter for reading the result:

- **Development split only.** Held-out is reserved for the single final report on
  the best development configuration (brief F3 / rule A1), so this diagnostic
  never touches it.
- **Stemming differs and is not "fixed".** Postgres `english` FTS stems; this
  offline BM25 does not. So this compares *ranking functions under different
  tokenization*, not a controlled single-variable swap (brief trap). Stated, not
  corrected — matching the stemmer would be a different study.
- **Pooling bias on topical only.** BM25 was not one of the four retrievers used
  to build the frozen topical judgment pools, so it can surface papers nobody
  judged. Those are scored grade 0 and recorded per row in `unjudged_ranked_ids`,
  which makes topical BM25 a *lower bound*. The pre-registered decision rule reads
  `lexical` and `paraphrase`, whose relevance is a fixed known-item set with no
  pool, so the rule itself is unaffected by this bias.

    python eval/tools/bm25_ablation.py
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from papertrail.database import session_scope
from papertrail.evaluation import (
    _aggregate_by_type,
    load_question_set,
    ndcg_at,
    recall_at,
    reciprocal_rank,
)
from papertrail.manifest import CorpusManifest
from papertrail.models import Chunk, Paper
from papertrail.retrieval import distinct_papers

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "evidence" / "corpus-manifest-1000.json"
QUESTIONS = ROOT / "eval" / "questions-v3.json"
OUT = ROOT / "docs" / "evidence" / "phase-8-bm25-offline.json"

# Exactly the token regex `repository.keyword_search` uses, plus lower-casing.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
MIN_QUERY_TOKEN_LEN = 3  # keyword_search drops query terms shorter than 3 chars

# rank_bm25 BM25Okapi defaults, recorded in the evidence rather than assumed.
K1 = 1.5
B = 0.75
EPSILON = 0.25

CANDIDATE_CHUNKS = 200
LIMIT = 10


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def tokenize_query(text: str) -> list[str]:
    """Query-side tokenization, matching keyword_search: same regex, lower-cased,
    terms shorter than 3 characters dropped, order-preserving de-duplication."""
    return list(
        dict.fromkeys(t for t in tokenize(text) if len(t) >= MIN_QUERY_TOKEN_LEN)
    )


def main() -> int:
    manifest = CorpusManifest.read(MANIFEST)
    question_set = load_question_set(QUESTIONS, manifest)

    with session_scope() as session:
        rows = session.execute(
            select(Chunk.id, Paper.arxiv_id, Paper.title, Paper.source_url, Chunk.ordinal, Chunk.text)
            .join(Paper, Paper.id == Chunk.paper_id)
            .order_by(Chunk.id)
        ).all()

    corpus_tokens = [tokenize(r.text) for r in rows]
    build_started = time.perf_counter()
    bm25 = BM25Okapi(corpus_tokens, k1=K1, b=B, epsilon=EPSILON)
    build_ms = (time.perf_counter() - build_started) * 1000

    pools = {
        q["id"]: set(q["pool"])
        for q in question_set["retrieval_questions"]
        if q.get("pool")
    }

    evaluated = [
        q for q in question_set["retrieval_questions"] if q["split"] == "development"
    ]

    out_rows: list[dict] = []
    for item in evaluated:
        graded = {str(k): int(v) for k, v in item["relevant"].items()}
        terms = tokenize_query(item["query"])
        started = time.perf_counter()
        scores = bm25.get_scores(terms) if terms else [0.0] * len(rows)
        # Top candidate chunks by BM25, deterministic on ties via chunk id.
        order = sorted(
            range(len(rows)), key=lambda i: (-float(scores[i]), int(rows[i].id))
        )[:CANDIDATE_CHUNKS]
        score_ms = (time.perf_counter() - started) * 1000
        hits = [
            {
                "arxiv_id": rows[i].arxiv_id,
                "title": rows[i].title,
                "source_url": rows[i].source_url,
                "chunk_id": int(rows[i].id),
                "ordinal": int(rows[i].ordinal),
                "text": rows[i].text,
                "score": float(scores[i]),
            }
            for i in order
        ]
        selected = distinct_papers(hits, limit=LIMIT)
        ranked = [str(h["arxiv_id"]) for h in selected]

        row = {
            "question_id": item["id"],
            "split": item["split"],
            "mode": "bm25_offline",
            "type": item["type"],
            "relevant_arxiv_ids": item["relevant_arxiv_ids"],
            "relevant": graded,
            "ranked_arxiv_ids": ranked,
            "query_terms": terms,
            "recall_at_5": recall_at(ranked, graded, 5),
            "recall_at_10": recall_at(ranked, graded, 10),
            "reciprocal_rank": reciprocal_rank(ranked, graded),
            "ndcg_at_10": ndcg_at(ranked, graded, 10),
            "top_score": float(selected[0]["score"]) if selected else 0.0,
            "latency_ms": score_ms,
        }
        pool = pools.get(item["id"])
        if pool is not None:
            row["unjudged_ranked_ids"] = sorted(set(ranked) - pool)
        out_rows.append(row)

    aggregates = {"development": {"bm25_offline": _aggregate_by_type(out_rows)}}

    report = {
        "schema_version": 1,
        "kind": "bm25_offline",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": question_set["frozen_at_utc"],
        "evaluation_schema_version": question_set["schema_version"],
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "protocol": {
            "library": f"rank_bm25 {version('rank_bm25')}",
            "ranker": "BM25Okapi",
            "k1": K1,
            "b": B,
            "epsilon": EPSILON,
            "chunks_indexed": len(rows),
            "candidate_chunks": CANDIDATE_CHUNKS,
            "retrieval_limit": LIMIT,
            "splits_evaluated": ["development"],
            "heldout_withheld_reason": (
                "Held-out is reserved for one final report on the best development "
                "configuration (brief F3, rule A1)."
            ),
            "tokenizer": "[A-Za-z0-9]+(?:-[A-Za-z0-9]+)* lower-cased; query terms <3 chars dropped",
            "stemming": (
                "none — Postgres english FTS stems and this does not, so the "
                "comparison is across different tokenization, not a controlled swap"
            ),
            "index_build_ms": build_ms,
            "comparison_baseline": "docs/evidence/phase-8-retrieval-v3-baseline.json (keyword mode, development)",
        },
        "aggregates": aggregates,
        "per_question": out_rows,
        "claim_boundary": (
            "Offline in-process BM25 over chunk text, development split only. "
            "Latencies are in-process scoring time and are NOT comparable to the "
            "SQL keyword path (A10). BM25 did not contribute to the frozen topical "
            "judgment pools, so topical scores are a lower bound: papers it surfaces "
            "that nobody judged are scored grade 0 and listed in unjudged_ranked_ids."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "rows": len(out_rows),
                      "aggregates": aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
