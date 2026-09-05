"""Sample papers uniformly at random from the corpus manifest for v3 labeling.

Deterministic by construction: the manifest IDs are sorted before sampling and
the selection uses ``random.Random(SEED)``, so the exact 90-paper reading set is
reproducible and the seed can be recorded in the frozen v3 question file.

Writes two files next to this script:
- ``sample.md``   — human-readable: id, title, abstract, for writing queries.
- ``sample.json`` — machine-readable: the sampled ids in sampled order.

Run from the repo with the DB up and the package importable::

    python eval/tools/sample_papers.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from sqlalchemy import select

from papertrail.database import session_scope
from papertrail.manifest import CorpusManifest
from papertrail.models import Paper

SEED = 20260903
SAMPLE_SIZE = 90

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "docs" / "evidence" / "corpus-manifest-1000.json"
OUT_MD = HERE / "sample.md"
OUT_JSON = HERE / "sample.json"


def sample_ids() -> list[str]:
    manifest = CorpusManifest.read(MANIFEST)
    ids = sorted(manifest.arxiv_ids)  # canonical order before sampling
    rng = random.Random(SEED)
    return rng.sample(ids, SAMPLE_SIZE)


def main() -> int:
    chosen = sample_ids()
    with session_scope() as session:
        rows = session.execute(
            select(Paper.arxiv_id, Paper.title, Paper.abstract).where(
                Paper.arxiv_id.in_(chosen)
            )
        ).all()
    by_id = {row.arxiv_id: (row.title, row.abstract) for row in rows}
    missing = [aid for aid in chosen if aid not in by_id]
    if missing:
        raise SystemExit(f"sampled ids missing from the database: {missing}")

    lines = [
        f"# v3 labeling sample — {SAMPLE_SIZE} papers (seed {SEED})",
        "",
        "One query per paper. Do NOT look at any retrieval output while writing "
        "queries. For paraphrase questions, no word from the title may appear in "
        "the query (enforced later by a script).",
        "",
    ]
    for index, aid in enumerate(chosen, start=1):
        title, abstract = by_id[aid]
        lines.extend(
            [
                f"## {index}. `{aid}`",
                "",
                f"**{title}**",
                "",
                abstract.strip(),
                "",
                "- type: ",
                "- query: ",
                "",
                "---",
                "",
            ]
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {"seed": SEED, "sample_size": SAMPLE_SIZE, "arxiv_ids": chosen}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seed": SEED,
                "sampled": len(chosen),
                "md": str(OUT_MD.relative_to(ROOT)),
                "json": str(OUT_JSON.relative_to(ROOT)),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
