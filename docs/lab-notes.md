# PaperTrail retrieval-upgrade lab notes

A running log of the retrieval-upgrade work. One dated entry per work session:
what I ran, what I saw, what surprised me. Interview answers come from here.

---

## 2026-09-03 — Phase S (Setup)

Environment note: two virtualenvs exist. `.venv` has PaperTrail but **not**
the ML extra; `.venv-ml` has PaperTrail plus `sentence-transformers 5.6.1`.
Used `.venv-ml` for this phase because it is the superset (covers both the
cross-encoder check and the baseline, which only needs the three
non-reranking modes). Config defaults already point the DB at
`localhost:5432`, matching the exposed Compose port, so no `.env` was needed;
retrieval evaluation needs no Groq key.

**S1 — database up and verified.**
- `docker compose up -d db` → container healthy.
- `alembic upgrade head` → already at head (the `papertrail-postgres` named
  volume persisted schema + data).
- `papertrail verify-manifest docs/evidence/corpus-manifest-1000.json` →
  `{"verified": true, "papers": 1000,
  "arxiv_ids_sha256": "7308d240f717df7c9b17a1dfb7140c68615a462298b42866b95bc17f93250146"}`.
  SHA matches the manifest, so the corpus is intact; no replay/embed needed.

**S2 — cross-encoder loads.**
- `CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')` loaded and predicted
  on `['what is RRF', 'Reciprocal rank fusion combines rankings']`.
- Score printed: **-11.000422**. A number, not an exception — that is all this
  step needs. (The raw ms-marco logit is not calibrated; a negative value here
  is not meaningful on its own, only relative to other pairs.)

**S3 — frozen baseline reproduces.**
- `papertrail evaluate --questions eval/questions-v2.json
  --manifest docs/evidence/corpus-manifest-1000.json
  --output /tmp/baseline-rerun.json`
- `papertrail verify-evidence --kind retrieval --report /tmp/baseline-rerun.json
  --manifest docs/evidence/corpus-manifest-1000.json` →
  `{"verified": true, "kind": "retrieval", "raw_rows": 90}`.
- Compared every aggregate against
  `docs/evidence/phase-6-retrieval-evaluation-v2.json`: **all metrics match
  exactly** (recall@5 / MRR / nDCG@10 across vector·keyword·hybrid for both
  dev and held-out, plus every abstention aggregate and the threshold).
  Latency differs (expected), aggregate metrics do not.

  Frozen values, for reference:

  | split | mode | recall@5 | MRR | nDCG@10 |
  |---|---|---|---|---|
  | dev | vector | 1.0000 | 1.0000 | 1.0000 |
  | dev | keyword | 0.9000 | 0.9063 | 0.9158 |
  | dev | hybrid | 1.0000 | 0.9417 | 0.9565 |
  | heldout | vector | 1.0000 | 1.0000 | 1.0000 |
  | heldout | keyword | 0.9000 | 0.8500 | 0.8856 |
  | heldout | hybrid | 1.0000 | 0.9500 | 0.9631 |

  Abstention: threshold 0.03239446668849102; dev balanced-acc 0.925, held-out
  0.900; negatives abstained 100% on both splits.

**What surprised me / to revisit:** the v2 set is saturated — vector nDCG@10 is
a flat 1.0 on both splits. That is exactly the motivation for Phase 0 (a
paraphrase / lexical / pooled-topical benchmark where different retrievers can
fail in different places). Also worth noting for Phase 1b: the abstention
positive scores cluster tightly around 0.0325–0.0328 and the threshold sits at
0.03239 — the rank-quantization the workbook warns about is already visible in
these numbers.

**Checkpoint S:** ✅ database verified, cross-encoder loads, baseline reproduces
exactly.
