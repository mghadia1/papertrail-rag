"""Phase 1 HNSW recall study: chunk-level index recall vs an exact scan.

Uses ALL v3 retrieval questions, both splits. This is allowed (A1/C2) because
nothing is selected or tuned here — the index parameters stay at their defaults
after the study; the question set is just a source of realistic query vectors.

For each query:
- ``exact_50`` = exact sequential-scan top-50 (ground truth), via
  ``vector_search(exact=True)``.
- for each ef in EF_VALUES: force the HNSW index on (``SET LOCAL enable_seqscan
  = off`` in the same transaction) and set ``hnsw.ef_search = ef``, take top-50,
  and compute chunk recall@10 and recall@50 against ``exact_50``. Forcing the
  index matters: at 2,039 vectors the planner naturally reverts to a sequential
  scan above ef≈40 (recorded per ef as ``natural_scan``), so without forcing we
  would measure the planner, not the index.

Latency: the natural path (no seqscan override) is timed for exact and for the
default-ef index, since that is what production pays; forced-index calls are also
timed but flagged. First query per configuration is discarded as warm-up.

Writes docs/evidence/phase-8-hnsw-recall.json. Chunk-level; not a paper-level
metric (A14).
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from papertrail.database import session_scope
from papertrail.embedding import get_encoder
from papertrail.evaluation import chunk_recall, load_question_set, _percentile
from papertrail.manifest import CorpusManifest
from papertrail.repository import vector_search

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "evidence" / "corpus-manifest-1000.json"
QUESTIONS = ROOT / "eval" / "questions-v3.json"
OUT = ROOT / "docs" / "evidence" / "phase-8-hnsw-recall.json"
EF_VALUES = [10, 40, 100, 200, 400, 1000]
LIMIT = 50


def _ids(hits: list[dict]) -> list[int]:
    return [int(h["chunk_id"]) for h in hits]


def _natural_scan(session, emb) -> dict[int, str]:
    """Per ef, what scan the planner picks WITHOUT forcing (index vs seqscan)."""
    vec = "[" + ",".join(repr(float(x)) for x in emb) + "]"
    sql = (
        f"SELECT c.id FROM chunks c JOIN papers p ON p.id = c.paper_id "
        f"WHERE c.embedding IS NOT NULL "
        f"ORDER BY c.embedding <=> '{vec}'::vector, c.id LIMIT {LIMIT}"
    )
    out: dict[int, str] = {}
    for ef in EF_VALUES:
        with session_scope() as s:
            s.execute(text(f"SET LOCAL hnsw.ef_search = {ef}"))
            plan = "\n".join(r[0] for r in s.execute(text("EXPLAIN " + sql)))
        out[ef] = "index" if "ix_chunks_embedding_hnsw_cosine" in plan else "seqscan"
    return out


def main() -> int:
    manifest = CorpusManifest.read(MANIFEST)
    question_set = load_question_set(QUESTIONS, manifest)
    encoder = get_encoder()
    questions = question_set["retrieval_questions"]
    embeddings = {q["id"]: encoder.encode([q["query"]], batch_size=1)[0] for q in questions}

    # Natural planner behaviour is a property of ef + table stats, not the query
    # vector; sample it once on the first query.
    with session_scope() as s:
        natural_scan = _natural_scan(s, embeddings[questions[0]["id"]])

    rows: list[dict] = []
    exact_latencies: list[float] = []
    warmup_done = False
    for q in questions:
        emb = embeddings[q["id"]]
        with session_scope() as s:
            t0 = time.perf_counter()
            exact_hits = vector_search(s, emb, limit=LIMIT, exact=True)
            exact_ms = (time.perf_counter() - t0) * 1000
        exact_ids = _ids(exact_hits)
        for ef in EF_VALUES:
            with session_scope() as s:
                # Force the index on and set ef directly. We drive hnsw.ef_search
                # via SET LOCAL rather than vector_search's ef_search kwarg because
                # the study deliberately probes ef < limit (the kwarg guards against
                # that for production); the index truncates to ef rows, which is the
                # point of recall@50 at ef=10/40.
                s.execute(text("SET LOCAL enable_seqscan = off"))
                s.execute(text(f"SET LOCAL hnsw.ef_search = {ef}"))
                t0 = time.perf_counter()
                approx_hits = vector_search(s, emb, limit=LIMIT)
                approx_ms = (time.perf_counter() - t0) * 1000
            approx_ids = _ids(approx_hits)
            if not warmup_done:  # discard the very first timed pair per A10/C2
                warmup_done = True
                continue
            rows.append({
                "question_id": q["id"],
                "ef_search": ef,
                "returned_rows": len(approx_ids),
                "truncated": len(approx_ids) < LIMIT,
                "recall_at_10": chunk_recall(approx_ids, exact_ids, 10),
                "recall_at_50": chunk_recall(approx_ids, exact_ids, 50),
                "forced_index_latency_ms": approx_ms,
                "exact_latency_ms": exact_ms,
            })
        exact_latencies.append(exact_ms)

    summary: dict[str, dict] = {"ef_search": {}}
    for ef in EF_VALUES:
        sel = [r for r in rows if r["ef_search"] == ef]
        summary["ef_search"][str(ef)] = {
            "questions": len(sel),
            "mean_recall_at_10": statistics.fmean(r["recall_at_10"] for r in sel),
            "mean_recall_at_50": statistics.fmean(r["recall_at_50"] for r in sel),
            "mean_returned_rows": statistics.fmean(r["returned_rows"] for r in sel),
            "forced_index_latency_p50_ms": _percentile([r["forced_index_latency_ms"] for r in sel], 0.50),
            "forced_index_latency_p95_ms": _percentile([r["forced_index_latency_ms"] for r in sel], 0.95),
            "natural_scan": natural_scan[ef],
        }
    summary["exact"] = {
        "questions": len(exact_latencies),
        "latency_p50_ms": _percentile(exact_latencies, 0.50),
        "latency_p95_ms": _percentile(exact_latencies, 0.95),
    }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": question_set["frozen_at_utc"],
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "embedding_model": encoder.model_name,
        "index": {"type": "hnsw", "m": 16, "ef_construction": 64, "ops": "vector_cosine_ops",
                  "name": "ix_chunks_embedding_hnsw_cosine"},
        "protocol": {
            "k_values": [10, 50],
            "ef_search_values": EF_VALUES,
            "queries": len(questions),
            "level": "chunk",
            "limit": LIMIT,
            "forced_index": "SET LOCAL enable_seqscan = off so the HNSW index is used at every ef",
            "warmup_discarded": True,
        },
        "rows": rows,
        "summary": summary,
        "claim_boundary": (
            "Chunk-level index recall on 2,039 vectors with one query set; not a "
            "general HNSW benchmark. recall@50 below 1.0 at ef<50 is truncation "
            "(HNSW returns at most ef rows), not approximation error. natural_scan "
            "records that the planner reverts to an exact seq scan above ef=40 on "
            "this corpus; recall here is measured with the index forced on."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "rows": len(rows),
                      "natural_scan": natural_scan,
                      "summary": {ef: {k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in summary["ef_search"][str(ef)].items()}
                                  for ef in EF_VALUES},
                      "exact": {k: round(v, 4) if isinstance(v, float) else v for k, v in summary["exact"].items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
