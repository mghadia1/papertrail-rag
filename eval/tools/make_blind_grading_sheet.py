"""Render the 18 topical judgment pools as a BLIND grading sheet.

No draft grades, no seed marker, no mode/rank — just the query and the shuffled
pool, so an independent grader assigns 0/1/2 without anchoring. Grades are then
compared to the Claude draft grades to report an inter-annotator disagreement
rate (score_blind_grades.py).

    python eval/tools/make_blind_grading_sheet.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POOLS = [HERE / "pools.json", HERE / "pools-heldout.json"]
DRAFT = ROOT / "eval" / "questions-v3.draft.json"
OUT = HERE / "topical-blind-grading.md"
SNIPPET = 320  # chars of abstract shown per paper


def main() -> int:
    # Index every pool by its seed; the two files reuse qids, so seed is the key.
    pool_by_seed: dict[str, dict] = {}
    for path in POOLS:
        if path.exists():
            for info in json.loads(path.read_text()).values():
                pool_by_seed[info["seed"]] = info

    # Only include pools for CURRENT topical questions (excludes discarded ones).
    draft = json.loads(DRAFT.read_text())
    topical = [q for q in draft["retrieval_questions"] if q["type"] == "topical"]

    lines = [
        "# Blind topical grading sheet",
        "",
        "For each query, mark every paper 0 / 1 / 2:",
        "- **2** = primary: squarely about the query topic.",
        "- **1** = partial: adjacent / touches the topic.",
        "- **0** = not relevant (leave blank).",
        "",
        "Write your grade in the `[ ]`. No draft grades or source papers are shown; "
        "order is shuffled. Full abstracts are in pools.json / pools-heldout.json.",
        "",
    ]
    total = 0
    missing = []
    for q in topical:
        info = pool_by_seed.get(q["seed"])
        if info is None:
            missing.append(q["id"])
            continue
        lines.append(f"\n## {q['id']} — {info['query']}\n")
        for p in info["pool"]:
            total += 1
            abstract = " ".join(p["abstract"].split())[:SNIPPET]
            lines.append(f"- [ ] `{p['arxiv_id']}` — **{p['title']}**")
            lines.append(f"      {abstract}...")
        lines.append("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "pools": len(topical) - len(missing),
                      "papers": total, "missing_pool": missing}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
