"""Sample FRESH held-out papers, disjoint from the original 90-paper dev sample.

The original held-out split was observed during a dry-run, so it is discarded and
re-authored from papers that were never in the dev sample (sample.json). Uses a
distinct seed and excludes every dev-sample id. No dry-run is run on these.

    python eval/tools/sample_papers_heldout.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from sqlalchemy import select

from papertrail.database import session_scope
from papertrail.manifest import CorpusManifest
from papertrail.models import Paper

SEED = 20260905  # distinct from the dev seed (20260903)
SAMPLE_SIZE = 40  # headroom over the 26 held-out seeds needed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "docs" / "evidence" / "corpus-manifest-1000.json"
DEV_SAMPLE = HERE / "sample.json"
OUT_MD = HERE / "sample-heldout.md"
OUT_JSON = HERE / "sample-heldout.json"


def main() -> int:
    manifest = CorpusManifest.read(MANIFEST)
    dev = set(json.loads(DEV_SAMPLE.read_text())["arxiv_ids"])
    pool = sorted(set(manifest.arxiv_ids) - dev)  # unsampled papers only
    rng = random.Random(SEED)
    chosen = rng.sample(pool, SAMPLE_SIZE)
    assert not (set(chosen) & dev), "held-out sample overlaps the dev sample"

    with session_scope() as session:
        rows = session.execute(
            select(Paper.arxiv_id, Paper.title, Paper.abstract).where(
                Paper.arxiv_id.in_(chosen)
            )
        ).all()
    by_id = {r.arxiv_id: (r.title, r.abstract) for r in rows}

    lines = [
        f"# v3 HELD-OUT labeling sample — {SAMPLE_SIZE} fresh papers (seed {SEED})",
        "",
        "Disjoint from the dev sample. Author held-out queries here; do NOT dry-run.",
        "",
    ]
    for index, aid in enumerate(chosen, start=1):
        title, abstract = by_id[aid]
        lines.extend([f"## {index}. `{aid}`", "", f"**{title}**", "", abstract.strip(),
                      "", "- type: ", "- query: ", "", "---", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(
        {"seed": SEED, "sample_size": SAMPLE_SIZE, "excluded_dev": len(dev), "arxiv_ids": chosen},
        indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seed": SEED, "sampled": len(chosen),
                      "disjoint_from_dev": not (set(chosen) & dev),
                      "md": str(OUT_MD.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
