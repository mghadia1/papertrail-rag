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

---

## 2026-09-03 — Phase H (Honesty fixes)

**H1 — reranker fallback made explicit** (`src/papertrail/reranking.py`).
- `CrossEncoderReranker.__init__` now takes `allow_fallback: bool = False`.
- `_load_model` re-raises a `RuntimeError` naming the model when the load fails
  and fallback is not allowed; only with `allow_fallback=True` does it drop to
  the lexical reranker, and it relabels `self.model_name` to the `-fallback`
  name so the substitution is visible.
- `rerank` stamps `item["reranker"] = self.model_name` on **every** row, in both
  the real and fallback paths.
- Tests: updated `test_cross_encoder_fallback_works_when_model_absent` to pass
  `allow_fallback=True` (and assert the row is stamped with the `-fallback`
  name); added `test_cross_encoder_default_raises_when_model_cannot_load`, which
  monkeypatches the `sentence_transformers` import to fail and asserts the
  default path raises. Both use an import-failure monkeypatch so they are
  offline and deterministic.

**H2 — `hybrid_rerank` exposed on the CLI** (`cli.py` + `retrieval.py`).
- Added `hybrid_rerank` to `search --mode` choices.
- Pointed `retrieve()`'s rerank default at `CrossEncoderReranker` (was the
  lexical reranker), so "rerank" means the real cross-encoder everywhere and
  raises loudly if the model is missing instead of silently simulating it.
- `papertrail search "diffusion planning" --mode hybrid_rerank` →
  `reranker=cross-encoder/ms-marco-MiniLM-L-6-v2` on every row (scores 0.6262,
  0.4801). The real model name shows.

**H3 — docs corrected.**
- README + `docs/status.md` test count: the files said **41** (workbook expected
  48); actual is now **49** after the new test. Updated both to 49.
- Added to both: "A `hybrid_rerank` mode and a statement-level NLI check exist in
  code; neither has a published metric." Verified this is honest —
  `src/papertrail/entailment.py` exists and no evidence file publishes a rerank
  or entailment metric.

**Environment gotcha (write this down):** `.venv-ml` uses a **non-editable**
install. A path-style editable `.pth` is not honored in this venv (the project
path contains a space — `auto job applier`), so `src/` edits do **not** take
effect until `pip install . --no-deps` is re-run. Two editable attempts
(default and `editable_mode=compat`) both failed to put `src` on `sys.path`;
`PYTHONPATH=src` works but reinstalling non-editable is the clean path.

**Verification:** `pytest` → **49 passed**. Post-edit, the frozen v2 retrieval
baseline still reproduces **exactly** (the three evaluated modes never touch the
rerank branch).

**Checkpoint H:** ✅ tests green, README/status accurate, `search --mode
hybrid_rerank` shows the real model name.

**Interview note:** "I found a silent reranker fallback that could have
mislabeled evidence — if the cross-encoder failed to load, it quietly scored
with a lexical stand-in under the same name. I made the fallback opt-in
(`allow_fallback`), made the default raise with the model name, and stamped the
model that actually ran onto every result row."
