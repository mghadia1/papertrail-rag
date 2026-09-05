"""Render the v3 draft question set as a human-readable review document.

    python eval/tools/make_review_doc.py   # writes eval/tools/v3-review.md
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from papertrail.database import session_scope
from papertrail.models import Paper

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRAFT = ROOT / "eval" / "questions-v3.draft.json"
OUT = HERE / "v3-review.md"


def main() -> int:
    payload = json.loads(DRAFT.read_text())
    rq = payload["retrieval_questions"]
    ab = payload["abstention_questions"]
    ids = {aid for q in rq for aid in q["relevant"]}
    with session_scope() as session:
        rows = session.execute(select(Paper.arxiv_id, Paper.title).where(Paper.arxiv_id.in_(ids))).all()
    title = {r.arxiv_id: r.title for r in rows}

    L: list[str] = []
    L.append("# v3 question set — review draft (NOT frozen)")
    L.append("")
    L.append(payload.get("authorship", ""))
    L.append(f"\nSampling seed {payload['sampling_seed']}. "
             f"{len(rq)} retrieval questions, {len(ab)} abstention questions.\n")

    for split in ("development", "heldout"):
        for qtype in ("paraphrase", "lexical", "topical"):
            group = [q for q in rq if q["split"] == split and q["type"] == qtype]
            L.append(f"\n## {split} · {qtype} ({len(group)})\n")
            for q in group:
                L.append(f"**{q['id']}** — {q['query']}")
                if qtype == "topical":
                    L.append(f"  - seed: `{q['seed']}` · pool size {len(q['pool'])}")
                    graded = sorted(q["relevant"].items(), key=lambda kv: (-kv[1], kv[0]))
                    for aid, g in graded:
                        L.append(f"  - grade {g}: `{aid}` — {title.get(aid, '?')}")
                else:
                    (aid,) = q["relevant"].keys()
                    L.append(f"  - answer: `{aid}` — {title.get(aid, '?')}")
                L.append("")

    L.append("\n## abstention — out-of-domain negatives\n")
    for q in ab:
        if q["type"] == "negative_ood":
            L.append(f"- ({q['split']}) {q['query']}")
    L.append("\n## abstention — near-miss negatives (real ML topics verified absent)\n")
    for q in ab:
        if q["type"] == "negative_near":
            L.append(f"- ({q['split']}) {q['query']}")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
