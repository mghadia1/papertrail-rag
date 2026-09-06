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

---

## 2026-09-05 — Phase 0 (Evaluation v3), part 1: assumptions + sampling

**0a — the three hard-coded assumptions I am about to break, confirmed in code:**
1. `evidence.verify_retrieval_evidence` hard-codes **90** raw rows
   (`evidence.py:35`), **20** dev / **10** held-out per mode (`:37`), and
   `protocol.rrf_k == 60` (`:64`).
2. `evaluation.MODES` is a fixed 3-tuple `("vector","keyword","hybrid")`
   (`evaluation.py:21`); `load_question_set` accepts only schema 1/2 and asserts
   exactly 30 retrieval questions / 20 dev / 10 held-out / 10+5 negatives
   (`:58`, `:76-100`).
3. The question file's relevance is a **flat `relevant_arxiv_ids` list with no
   grades** (`evaluation.py:86,174`), and `ndcg_at` uses **binary** gains
   (`:35-45`). There is no `recall_at(...,10)` recorded (only `recall_at_5`).

**0b — deterministic sampler** (`eval/tools/sample_papers.py`).
- Sorts the manifest IDs, then `random.Random(20260903).sample(ids, 90)`.
- Joins title + abstract from the DB, writes `eval/tools/sample.md` (a reading
  sheet with a `type:`/`query:` slot per paper) and `eval/tools/sample.json`
  (the 90 ids in sampled order, for the freeze to reference).
- Ran it: 90 papers, seed 20260903. Corpus is August-2026 arXiv ML papers
  (ids `2608.*`).

**0c–0e — authored the draft (2026-09-05), pending human review.**
- 78 retrieval queries + 27 negatives → `eval/questions-v3.draft.json`
  (`frozen_at_utc` left null). Counts match the workbook: paraphrase 24/12,
  lexical 16/8, topical 12/6; OOD negatives 15 (reused), near-miss 12.
- All 36 paraphrase queries pass the strict no-title-word checker.
- 12 near-miss negatives verified absent from the corpus by grep (sarcasm,
  fake-news, hate-speech, crowd-counting, lane-detection, sound-event, stock,
  load-forecasting, traffic-signal RL, drone-racing, crop-yield, handwriting).
- 18 topical pools built mode-blind and shuffled (`pool.py` → `pools.json`,
  ~680 papers) and graded by an LLM (Claude), disclosed. Every graded id lies in
  its judged pool; the full pool is stored per question for the verifier.
- Tooling: `build_v3.py`, `check_no_title_words.py`, `pool.py`,
  `make_review_doc.py`.

**Dry-run on the draft (not frozen), 4 modes.** Environment note: Docker Desktop
had shut down between sessions, dropping the DB mid-run once ("server closed the
connection unexpectedly"); relaunched Docker, re-verified the manifest
(`7308d240…`), re-ran clean.

Per-type nDCG@10 (dev): vector all **0.874** (was a flat 1.00 in v2), paraphrase
0.883, topical **0.687**, lexical 1.000. Recall@5 (dev): vector paraphrase 0.958,
lexical 1.000, topical 0.917; keyword paraphrase **0.750**; hybrid_rerank 1.000
across the board. Abstention balanced-accuracy fell to 0.828 dev / 0.810 held-out
(v2 was 0.925/0.90) — the near-miss negatives are genuinely harder.

Reading: the set is **no longer saturated** and retrievers fail in different
places (keyword dies on paraphrase; vector/hybrid weak on topical; rerank wins).
Checkpoint 0's explicit gate — dev paraphrase **or** lexical vector Recall@5
below 0.9 — is **not quite met** (0.958 / 1.000); the discrimination shows in
nDCG, not that coarse recall gate. Handed the draft to Mayank to review the query
wording and topical grades before any hardening or freeze (his call, 2026-09-05).
Nothing frozen yet.

**Mayank's review (2026-09-05) and the revisions I applied.** He read the draft,
tooling, notes, and commit. Verdict: do **not** harden queries (that gate was his
heuristic and would tune the benchmark to a specific retriever); the real problem
is that the **held-out split was observed** in the dry-run. Actions taken:
- **Do not harden.** He replaced Checkpoint 0's wording in the workbook with a
  per-type nDCG condition; the 0.958 stands.
- **Held-out replaced.** Discarded the 26 held-out questions + 9 held-out
  negatives that were dry-run, and re-authored them from a **fresh 40-paper
  sample disjoint from dev** (`sample_papers_heldout.py`, seed 20260905). No
  dry-run is run on the new held-out. Dev split (52 Q + 18 neg) kept.
- **Lexical rewritten.** The keyword-soup lexical queries are a form problem
  visible without results, so all 16 dev lexical (and the 8 fresh held-out
  lexical) are now single natural sentences carrying one or two rare terms.
- **Near-miss absence re-confirmed by grep** for load-forecasting, sarcasm,
  stock (dev) and protein-structure, homomorphic-encryption, gravitational-wave,
  exoplanet (held-out): all 0 hits across the 1,000 titles+abstracts.
- **Dry-run preserved** to `docs/evidence/phase-8-retrieval-v3-DRAFT-unfrozen.json`.
- **Claim boundary** now records that paraphrases are long, abstract-style
  rewrites (easier for dense retrieval than terse user queries).
- Held-out topical pooled fresh (`pool.py --ungraded` → `pools-heldout.json`,
  229 papers) and graded.
- **Blind re-grade + adjudication + freeze (2026-09-05).** Mayank had a second,
  independent Claude session (**Claude Fable 5.1**) grade the 18 topical pools
  blind from the shuffled sheet (`blind-grades.json`). Recomputed agreement with
  `score_blind_grades.py`: **Cohen kappa 0.7186** over 690 pooled papers (0.7089
  over the 573 papers in the 15 fully-blind pools; three pools — v3q025, v3q032,
  v3q050 — were not fully blind for that grader, noted in the JSON). This is a
  Claude-vs-Claude second opinion, **not** human inter-annotator agreement, and
  is labeled that way. **53 papers disagreed.** I adjudicated **every one by
  hand** against its abstract (`eval/tools/adjudication.md`): the draft had been
  too generous on generic-RAG (v3q054: 11→4 relevant), SLAM/driving (v3q073:
  5→2), and general-SR (v3q050); the blind grader caught two papers the draft
  missed (CheMatE 2608.03855 — abstract literally says "catastrophically
  forgetting"; self-distilled reward shaping 2608.03223) and one upgrade
  (2608.03745). Final labels are the adjudicated ones; the raw kappa is recorded
  in the frozen file's `topical_grade_provenance` and the evidence protocol.
- **FROZEN** `eval/questions-v3.json` at 2026-09-05T18:52:02Z (renamed from the
  draft; `frozen_at_utc` set).

**Checkpoint 0 — closed (2026-09-05).** Official 0g run on the frozen set →
`docs/evidence/phase-8-retrieval-v3-baseline.json`; `verify-evidence
--kind retrieval --questions eval/questions-v3.json` → **verified, 312 rows**
(78×4 modes). The schema-3 verifier recomputes every per-row metric from ranked
ids + grades, checks each row's coverage against the question file, confirms each
topical row's ranked ids stay inside its judged pool, and recomputes per-type
aggregates.

Frozen per-type nDCG@10 (dev): vector all **0.876**, topical **0.699**,
paraphrase 0.883, lexical 1.000; keyword paraphrase **0.695**; hybrid_rerank all
0.942. Held-out: vector topical **0.776** vs hybrid topical **0.683** —
**fusion hurt on held-out topical**, and keyword topical collapsed to 0.497.
Abstention balanced accuracy 0.828 dev / 0.846 held-out (v2 was 0.925/0.90; the
near-miss negatives are the reason). The revised Checkpoint 0 condition (per-type
nDCG discrimination, not the coarse Recall@5 gate) is met: v2 was a flat 1.00 on
every vector cell; v3 is not, and the four retrievers separate by type.

**Interview notes.**
Honesty note (A16): the labels were produced by models, not by me. These lines
are only defensible once I (Mayank) have actually read `eval/tools/adjudication.md`
and can argue the calls; until then they describe what the tooling did, not what
I did.
1. "My first benchmark was title-derived and saturated — every retriever scored
   1.00. It was rebuilt with paraphrase (no title words), lexical, and pooled
   topical queries so retrievers fail in different places: keyword dies on
   paraphrase, and on the 6 held-out topical questions RRF fusion scored below
   plain vector."
2. "Grades came from two independent model gradings that agreed at kappa 0.72;
   the draft model then adjudicated all 53 disagreements per abstract. The raw
   kappa is on record and it is labelled model-vs-model, not human, agreement.
   I reviewed the adjudication record before standing behind the labels."
   (Only say the last sentence once it is true.)
3. "A dry-run had touched the held-out split, so it was thrown out and
   re-authored from a fresh disjoint sample, the way v1's observed held-out was
   replaced for v2."

**0f — harness generalized for schema 3 (done before authoring, all green).**
Everything is schema-branched so the frozen v2/v1 evidence keeps verifying
byte-for-byte; only schema-3 sets get the new behavior.
- `evaluation.py`: `MODES` is now the 4-tuple (adds `hybrid_rerank`), with
  `LEGACY_MODES` = the original three used for schema ≤2. `ndcg_at` is graded
  (`gain = 2**grade - 1`) and reduces to the old binary formula when handed a
  set, so `test_retrieval_metrics_have_known_answers` still passes. Schema-3
  rows carry `type`, graded `relevant`, and `recall_at_10`; aggregates nest as
  `aggregates[split][mode]["all"]` and `[...][<type>]`.
- `load_question_set` accepts schema 3: graded `relevant` maps, typed
  questions, grades validated ∈ {1,2}, relevant ids in corpus, pool ⊇ graded
  ids, counts **derived** from the file (no fixed 30/20/10).
- `evidence.py`: the retrieval verifier branches on `evaluation_schema_version`.
  Schema ≤2 keeps the fixed 90/20/10 flat check. Schema 3 recomputes **every
  per-row metric** from ranked ids + grades, derives coverage from the
  `--questions` file, checks each topical row's ranked ids ⊆ its judged `pool`,
  and recomputes the per-type + "all" aggregates. CLI now passes `--questions`
  into the retrieval verifier.
- Tests: added graded-nDCG known answer (a=2,b=1 ranked [b,a] → 0.7967) and a
  schema-3 loader round trip + grade-range rejection. `pytest` → **51 passed**.
- Regression: the frozen **v2 and v1** retrieval evidence still verify (with and
  without `--questions`).

---

## 2026-09-05 — Part B + Phase 1: HNSW recall study

Working from the execution brief (`docs/retrieval-upgrade-brief.md`), which
supersedes the workbook.

**Pre-run (A19):** git on 8f18e1e; DB healthy, manifest verified (1000 IDs,
sha 7308d240…), 2039/2039 chunks embedded; package reinstalled (non-editable —
the editable `.pth` is not honored because the repo path contains a space);
`pytest` **51 passed**; frozen v2 (90 rows) and v3 (312 rows) evidence verify.

**Part B — attribution + qualification fixes (no evidence change).**
- Corrected the adjudication attribution everywhere in the tracked docs: it was
  the **draft model (Claude Opus 4.8)** re-deciding, not "a third model" and not
  a human. Removed "by hand"/"hand-adjudicated" from README, `results.md`,
  `adjudication.md`, and the lab-notes interview lines (A16: those read as
  *human* work).
- Qualified the held-out topical finding with its count ("the 6 held-out topical
  questions") in README and `results.md` (A11, B2).
- Fixed the interview note so it does not have Mayank claim he adjudicated; the
  models did, and the line is only true once he has read `adjudication.md`.
- **Frozen-file caveat:** `eval/questions-v3.json` and
  `phase-8-retrieval-v3-baseline.json` still carry the earlier
  `topical_grade_provenance.method` phrase "then hand adjudication…". They are
  frozen (A2) / evidence (A3) and are **not** edited; the same block names
  `adjudicator: "Claude Opus 4.8"`, so it is attributed to the model, not a
  human. `build_v3.py` source was corrected for any future regen.

**C1 — `vector_search` gains `exact` and `ef_search`** (`repository.py`), with a
docstring stating the A12/A13 facts. Verified plans by EXPLAIN ANALYZE on the
production ORM query (`ORDER BY embedding <=> q, chunks.id LIMIT 50`):
- `exact=True` (indexscan+bitmapscan off) → **Seq Scan on chunks** (2039 rows).
- default `ef_search=40` → **Index Scan using ix_chunks_embedding_hnsw_cosine**
  (Nested Loop + Incremental Sort), returns 40 rows.
- `ef_search=100` (natural) → the planner **reverts to Seq Scan** — the ANN index
  is only chosen at the low default ef on this 2,039-row table.
`SHOW hnsw.ef_search` read back the SET LOCAL value in-transaction (A12).

**C2 — study** (`eval/tools/hnsw_study.py`), all 78 v3 query vectors, index forced
on so recall measures the index not the planner. Table (from the JSON, A4):

| ef | recall@10 | recall@50 | rows | forced-idx p50 ms | natural scan |
|---|--:|--:|--:|--:|---|
| 10 | 0.957 | 0.200 | 10 | 3.6 | index |
| 40 | 0.990 | 0.800 | 40 | 4.2 | index |
| 100 | 0.999 | 0.998 | 50 | 4.6 | seqscan |
| 200 | 1.000 | 1.000 | 50 | 4.9 | seqscan |
| 400 | 1.000 | 1.000 | 50 | 5.5 | seqscan |
| 1000 | 1.000 | 1.000 | 50 | 6.7 | seqscan |

Exact scan p50 23.1 / p95 34.5 ms (n=78). Evidence
`docs/evidence/phase-8-hnsw-recall.json` (verified, 467 rows); C3 verifier
`--kind hnsw` recomputes every summary value from rows and enforces the
truncated-flag rule; C4 tests (chunk_recall, ef_search guard, verifier edit) →
**54 passed**.

**Design choice:** measured the index with `enable_seqscan=off` forced on, rather
than the planner's natural path, because at 2,039 vectors the planner declines the
index above ef=40 — forcing it is the only way to get a true recall@ef curve.

**Failure mode found:** the default `hnsw.ef_search=40` is **below** the retrieval
candidate pool (`candidate_limit` 50–200), so the production vector side is
silently capped at 40 candidates. Recorded, not fixed (A7); flagged for Phase 4.

**Surprises / not measured:** the planner reverting to an exact scan above ef=40
was unexpected and is the real answer to "is an ANN index worth it at 2k vectors"
— not yet. C5 index-parameter sweep **skipped**: recall@50 at ef=100 is 0.998
(≥0.99), the skip condition. Latencies include per-call connection setup, so they
are within-file relative only (A10).

**Post-run:** tests 54 passed; frozen v2 (90 rows) and v3 (312 rows) evidence
still verify; new `phase-8-hnsw-recall.json` verifies.

---

## 2026-09-06 — Phase 1b (D1–D4): abstention gate study

Pre-run: 54→ tests passing; DB up, manifest verified; package reinstalled; v2/v3/
hnsw evidence verify.

**D1** `eval/tools/gate_signals.py` → `docs/evidence/phase-8-gate-signals.json`:
8 candidate signals for all 105 v3 questions (78 answerable, 27 negatives),
both splits. Reranker that ran = `cross-encoder/ms-marco-MiniLM-L-6-v2` (asserted
not a fallback, A6). Counts: dev 52 answerable / 18 neg; held-out 26 answerable /
5 ood / 4 near.

**D2/D3** `eval/tools/score_gate.py` → `docs/evidence/phase-8-gate-selection.json`
(verified, `--kind gate`). Development AUROC and the held-out picture:

| signal | dev AUROC | dev balAcc | HO false-refuse (answerable, /26) | HO false-answer near (/4) | HO balAcc |
|---|--:|--:|--:|--:|--:|
| rrf_top (current) | 0.876 | 0.828 | 0.308 (8) | 0.000 (0) | 0.846 |
| cos_top | 0.993 | 0.971 | 0.077 (2) | 0.500 (2) | 0.850 |
| cos_margin | 0.765 | 0.725 | 0.308 | 0.750 | 0.568 |
| **cos_mean_top3** (chosen) | **0.998** | 0.981 | 0.038 (1) | 0.500 (2) | 0.870 |
| kw_top | 0.947 | 0.894 | 0.231 | 0.000 | 0.885 |
| ce_top | 0.904 | 0.850 | 0.077 | 0.500 | 0.850 |
| ce_margin | 0.811 | 0.784 | 0.154 | 0.000 | 0.923 |
| ce_sigmoid_top | 0.904 | 0.850 | 0.077 | 0.500 | 0.850 |

**The rank-quantization bug, in plain words:** `rrf_top` is a sum of `1/(k+rank)`
over the lists a paper appears in, so it can take only a few discrete values near
the threshold. A paper the vector search ranks #1 but keyword ranks low (or
misses) gets a *lower* `rrf_top` than a paper ranked ~1 in both, even though it
was clearly retrieved. Headline: **the current `rrf_top` gate refuses 8 of 26
answerable held-out questions whose relevant paper sat at hybrid rank 1 or 2** —
they were retrieved and then refused on score quantization, not on missing
evidence.

**Design choice:** selection is by dev AUROC (D2), which picks `cos_mean_top3`
(0.998) — the mean of the top-3 vector cosines, a smooth continuous score with
none of RRF's rank steps.

**Failure mode / the D8 trade-off (why I am stopping):** `cos_mean_top3` fixes
the false refusals (0.038 vs 0.308) and lifts held-out balanced accuracy
(0.870 vs 0.846), **but it answers 2 of the 4 held-out near-miss negatives**
(false-answer near 0.500 vs rrf_top's 0.000). Per brief rule D8, a default change
is allowed only if it does not raise the near-miss false-answer rate; this trades
one for the other, so I STOP and present. Also notable: `ce_margin` has the best
held-out balanced accuracy (0.923) and never answers a near-miss (0.000), but its
dev AUROC (0.811) lost the AUROC-only selection — a sign the selection metric
does not see the near-miss asymmetry. Small n on held-out (A11): near = 4.

**Post-run:** 56 tests pass; v2/v3/hnsw evidence still verify; gate evidence
verifies. No gate code (D5) or default (D8) changed — awaiting Mayank.

**Phase 1b D5+D6 (2026-09-06): configurable gate + soft abstention (default kept).**
Mayank's decision on the D3 table: keep `rrf_top`, do D5+D6 only, no default flip
(`cos_mean_top3` fails the near-miss rule and near n=4 is noisy).
- **D5:** new `papertrail/gate.py` holds `signal_from_hits` (the eight signals),
  per-signal `SIGNAL_RANGES`, `validate_threshold`, and `gate_signal`. Both the
  study (`gate_signals.py`, refactored to call `signal_from_hits`) and production
  (`answer_question`) now compute the identical number. `config.abstain_signal`
  added (default `"rrf_top"`); `answer_question` gained `gate_signal_name`,
  computes the configured signal instead of `hits[0]["score"]`, and validates the
  threshold against that signal's range (a logit gate has no [0,1] bound — the
  old hardcoded check moved into the signal definition). `AnswerResult` gained
  `gate_signal_name` and `gate_score`. **Default unchanged**, so all evidence
  still verifies and behavior is identical for `rrf_top`.
- **D6:** soft abstention — on a refusal `AnswerResult.nearest_papers` holds the
  top-3 retrieved (id, title, url) and no generation call is made; the API adds
  `nearest_papers` (+ `NearestPaper`), the CLI prints "Low confidence. Closest
  evidence:" to stderr while keeping stdout JSON.
- Tests: `signal_from_hits`/range checks, soft-abstention returns 3 nearest with
  `answer is None` and no generator call, non-abstention has empty `nearest_papers`;
  updated the threshold-validation message test. **59 passed.**
- **STOP before D7** (two-gate RAG eval) — it needs `GROQ_API_KEY`, which is not
  set here (confirmed: `ask` can't even build the generator without it).

**Phase 1b D7 (2026-09-06): BLOCKED by a decommissioned Groq model.**
Ran the three held-out RAG evaluations after the key was provided. Every
generation call returned **HTTP 404** — the configured `groq_model`
`llama-3.3-70b-versatile` is no longer served by Groq (confirmed: the key works,
`GET /v1/models` returns 200 but the list no longer contains it). The three runs
are preserved as `phase-8-rag-*-model404.json` (uncommitted); the harness'
error-capture worked (18/27/27 provider errors recorded, not silently dropped —
A6/A5).
- **The gate half of D7 is still valid** because the gate decides before any
  generation call: run 1 (`rrf_top`, prod threshold) refused **8** answerable
  held-out at hybrid rank 1–2 and abstained on all negatives; run 2
  (`cos_mean_top3`) refused **1** and answered **2/4** near-miss negatives —
  end-to-end confirmation of the D3 table.
- **The generation half is blocked**: answerable_answer_rate 0.0, citation
  grounding 0.0, entailment_refusals 0 — all artifacts of the 404, not findings.
- **Broader operational finding:** papertrail's live `ask`/`/ask` generation is
  currently non-functional against Groq until the model is updated. Available
  chat models now include `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
  `qwen/qwen3.8-27b`, `qwen/qwen3.6-27b`, `groq/compound`.
- `docs/status.md` and `PROJECT_SPEC.md` still claim "Groq/Llama 3.3 70B" — that
  claim is now stale and must change with the model.
- STOP for a model decision (config default + portfolio claim change, A18).
