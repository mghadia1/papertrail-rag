"""Part H cost measurement: one run, every embedding column, same machine state.

Phase 5 first reported embed wall times, index sizes and encode throughput from
terminal output, which A4 does not allow. This script re-measures them into a
verified evidence file, and — because all five encoders are timed back to back in
one process — the numbers are comparable to each other (A10), which the original
per-model embed runs on different days were not. MiniLM is included this time.

Per column, from its ``embedding_runs`` row (so each model uses the prefixes it
was indexed with):

* passage encode: every chunk text through ``encode_passage``, batch 32, after a
  discarded 32-chunk warm-up. Encode only — nothing is written to the database.
* HNSW index size: ``pg_relation_size`` of the column's index.
* query latency: each development question, ``encode()`` then ``vector_search``
  (limit 10, ef_search at the server default), after one discarded warm-up.
  Encode and search are recorded separately per row.

    python eval/tools/embed_costs.py
"""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from papertrail.database import session_scope
from papertrail.embedding import SentenceTransformerEncoder
from papertrail.evaluation import _percentile, load_question_set
from papertrail.manifest import CorpusManifest
from papertrail.repository import require_vector_search_ready, vector_search

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "evidence" / "corpus-manifest-1000.json"
QUESTIONS = ROOT / "eval" / "questions-v3.json"
OUT = ROOT / "docs" / "evidence" / "phase-8-embed-costs.json"

COLUMNS = (
    "embedding",
    "embedding_bge_small",
    "embedding_gte_small",
    "embedding_e5_small",
    "embedding_bge_base",
)
BATCH_SIZE = 32
LIMIT = 10


def summarize(passage_seconds: float, chunks: int, rows: list[dict]) -> dict:
    total = [r["encode_ms"] + r["search_ms"] for r in rows]
    return {
        "passage_encode_seconds": passage_seconds,
        "passage_chunks_per_second": chunks / passage_seconds,
        "query_encode_ms_p50": _percentile([r["encode_ms"] for r in rows], 0.50),
        "vector_search_ms_p50": _percentile([r["search_ms"] for r in rows], 0.50),
        "vector_query_ms_p50": _percentile(total, 0.50),
        "vector_query_ms_p95": _percentile(total, 0.95),
    }


def main() -> int:
    manifest = CorpusManifest.read(MANIFEST)
    question_set = load_question_set(QUESTIONS, manifest)
    items = [q for q in question_set["retrieval_questions"] if q["split"] == "development"]

    with session_scope() as session:
        texts = [row[0] for row in session.execute(text("SELECT text FROM chunks ORDER BY id"))]
        runs = {
            row.column_name: row
            for row in session.execute(text(
                "SELECT model_name, column_name, dimensions, query_prefix, passage_prefix "
                "FROM embedding_runs WHERE status = 'complete'"
            ))
        }

    columns: dict[str, dict] = {}
    for column in COLUMNS:
        run = runs[column]
        encoder = SentenceTransformerEncoder(
            run.model_name, dimensions=run.dimensions,
            query_prefix=run.query_prefix or "", passage_prefix=run.passage_prefix or "",
        )
        encoder.encode_passage(texts[:BATCH_SIZE], batch_size=BATCH_SIZE)
        started = time.perf_counter()
        encoder.encode_passage(texts, batch_size=BATCH_SIZE)
        passage_seconds = time.perf_counter() - started

        rows = []
        with session_scope() as session:
            require_vector_search_ready(
                session, model_name=run.model_name, dimensions=run.dimensions, column=column
            )
            index_bytes = session.execute(
                text("SELECT pg_relation_size(CAST(:name AS regclass))"),
                {"name": f"ix_chunks_{column}_hnsw_cosine"},
            ).scalar_one()
            warm = encoder.encode([items[0]["query"]], batch_size=1)[0]
            vector_search(session, warm, limit=LIMIT, column=column)
            for item in items:
                started = time.perf_counter()
                embedding = encoder.encode([item["query"]], batch_size=1)[0]
                encoded = time.perf_counter()
                vector_search(session, embedding, limit=LIMIT, column=column)
                finished = time.perf_counter()
                rows.append({
                    "question_id": item["id"],
                    "encode_ms": (encoded - started) * 1000,
                    "search_ms": (finished - encoded) * 1000,
                })
        columns[column] = {
            "model_name": run.model_name,
            "dimensions": run.dimensions,
            "query_prefix": run.query_prefix or "",
            "passage_prefix": run.passage_prefix or "",
            "hnsw_index_bytes": int(index_bytes),
            "summary": summarize(passage_seconds, len(texts), rows),
            "per_question": rows,
        }
        print(json.dumps({"column": column, **columns[column]["summary"],
                          "hnsw_index_bytes": int(index_bytes)}))
        del encoder

    report = {
        "schema_version": 1,
        "kind": "embed_costs",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": question_set["frozen_at_utc"],
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
        "chunk_count": len(texts),
        "protocol": {
            "columns": list(COLUMNS),
            "passage_batch_size": BATCH_SIZE,
            "passage_warmup_chunks_discarded": BATCH_SIZE,
            "passage_encode_scope": "encode only; no database writes",
            "query_split": "development",
            "query_warmups_discarded_per_column": 1,
            "vector_search_limit": LIMIT,
            "ef_search": "server default",
            "run": "all columns back to back in one process (A10)",
            "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        },
        "columns": columns,
        "claim_boundary": (
            "One local run on one machine. Latencies and throughput are comparable "
            "across columns within this file only (A10); they are not production "
            "latencies. Passage encode time excludes database writes, so it is not "
            "the full wall time of `papertrail embed`."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "columns": len(columns)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
