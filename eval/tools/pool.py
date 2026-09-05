"""Build judgment pools for the schema-3 topical questions.

For every topical question, retrieve in all four modes and union the returned
papers. We pool at both the pooling depth (limit=20, for judgment completeness)
and the evaluation depth (limit=10, so the frozen pool is guaranteed to contain
every id the evaluation can rank). Papers are shuffled with the sampling seed and
written with NO mode or rank shown, so grading is not biased by retriever order.

Outputs eval/tools/pools.json:
    {qid: {"query", "seed", "pool": [{"arxiv_id","title","abstract"}, ...]}}

    python eval/tools/pool.py            # reads eval/questions-v3.draft.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from sqlalchemy import select

from papertrail.database import session_scope
from papertrail.embedding import get_encoder
from papertrail.models import Paper
from papertrail.retrieval import retrieve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRAFT = ROOT / "eval" / "questions-v3.draft.json"
OUT = HERE / "pools.json"
SEED = 20260903
MODES = ("vector", "keyword", "hybrid", "hybrid_rerank")


def main() -> int:
    payload = json.loads(DRAFT.read_text(encoding="utf-8"))
    topical = [q for q in payload["retrieval_questions"] if q["type"] == "topical"]
    encoder = get_encoder()
    rng = random.Random(SEED)
    pools: dict[str, dict] = {}
    with session_scope() as session:
        for q in topical:
            ids: set[str] = set()
            for mode in MODES:
                enc = None if mode == "keyword" else encoder
                for limit in (20, 10):  # 10 guarantees the eval depth is covered
                    for hit in retrieve(session, q["query"], mode=mode, limit=limit, encoder=enc):
                        ids.add(str(hit["arxiv_id"]))
            ordered = sorted(ids)
            rng.shuffle(ordered)
            rows = session.execute(
                select(Paper.arxiv_id, Paper.title, Paper.abstract).where(
                    Paper.arxiv_id.in_(ordered)
                )
            ).all()
            by_id = {r.arxiv_id: (r.title, r.abstract) for r in rows}
            pools[q["id"]] = {
                "query": q["query"],
                "seed": q.get("seed"),
                "pool": [
                    {"arxiv_id": aid, "title": by_id[aid][0], "abstract": by_id[aid][1]}
                    for aid in ordered
                ],
            }
    OUT.write_text(json.dumps(pools, indent=2) + "\n", encoding="utf-8")
    sizes = {qid: len(p["pool"]) for qid, p in pools.items()}
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "questions": len(pools),
                      "pool_sizes": sizes, "total": sum(sizes.values())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
