"""Score an independent blind topical grading against the Claude draft grades.

Input: eval/tools/blind-grades.json, a map {qid: {arxiv_id: grade}} holding the
independent grader's non-zero grades (anything omitted is grade 0). qids and ids
must come from topical-blind-grading.md. The draft grades and the judged pool are
read from eval/questions-v3.draft.json.

Reports, over the union of each question's judged pool:
- 3-class exact-label agreement / disagreement (grades 0/1/2)
- binary relevant-vs-not agreement (grade>=1)
- Cohen's kappa (3-class), overall
plus a per-question table. Writes eval/tools/blind-grade-disagreement.json.

    python eval/tools/score_blind_grades.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRAFT = ROOT / "eval" / "questions-v3.draft.json"
BLIND = HERE / "blind-grades.json"
OUT = HERE / "blind-grade-disagreement.json"


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 1.0
    n = len(pairs)
    labels = (0, 1, 2)
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def main() -> int:
    draft = json.loads(DRAFT.read_text())
    if not BLIND.exists():
        raise SystemExit(
            f"missing {BLIND.relative_to(ROOT)} — provide {{qid: {{arxiv_id: grade}}}} "
            "from the independent grader first"
        )
    blind = json.loads(BLIND.read_text())

    per_q = []
    all_pairs: list[tuple[int, int]] = []
    for q in draft["retrieval_questions"]:
        if q["type"] != "topical":
            continue
        pool = q["pool"]
        d_grades = {k: int(v) for k, v in q["relevant"].items()}
        b_grades = {k: int(v) for k, v in blind.get(q["id"], {}).items()}
        pairs = [(d_grades.get(a, 0), b_grades.get(a, 0)) for a in pool]
        all_pairs.extend(pairs)
        exact = sum(1 for a, b in pairs if a == b)
        binary = sum(1 for a, b in pairs if (a >= 1) == (b >= 1))
        per_q.append({
            "id": q["id"],
            "pool": len(pool),
            "exact_agreement": round(exact / len(pairs), 4),
            "binary_agreement": round(binary / len(pairs), 4),
            "draft_relevant": sum(1 for v in d_grades.values() if v >= 1),
            "blind_relevant": sum(1 for v in b_grades.values() if v >= 1),
        })

    n = len(all_pairs)
    exact = sum(1 for a, b in all_pairs if a == b)
    binary = sum(1 for a, b in all_pairs if (a >= 1) == (b >= 1))
    summary = {
        "papers": n,
        "exact_label_agreement": round(exact / n, 4),
        "exact_label_disagreement": round(1 - exact / n, 4),
        "binary_relevance_agreement": round(binary / n, 4),
        "binary_relevance_disagreement": round(1 - binary / n, 4),
        "cohen_kappa_3class": round(cohen_kappa(all_pairs), 4),
        "per_question": per_q,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
