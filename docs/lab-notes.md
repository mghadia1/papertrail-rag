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

**Phase 1b D7 (2026-09-06): two-gate RAG evaluation, rerun on openai/gpt-oss-120b.**
Mayank picked `openai/gpt-oss-120b` to replace the decommissioned Llama-3.3.
Set via `.env` override for the runs (the config-default change is a separate
commit, A7). One smoke call confirmed grounded, cited output first. Three
held-out runs, all verified (`--kind rag`, 35 records each):

| gate (threshold) | answered/26 | refuse@rank1-2 | ood abstain | near abstain | entail refusals | no-cite refusals | grounding |
|---|--:|--:|--:|--:|--:|--:|--:|
| rrf_top (0.0324, current) | 13 (0.50) | 8 | 1.00 | 1.00 | 0 | 5 | 1.00 |
| cos_mean_top3 (0.4628) | 17 (0.65) | 1 | 0.78 | 0.50 | 0 | 10 | 1.00 |
| cos_mean_top3 + entailment | 0 (0.00) | 15 | 0.78 | 0.50 | 15 | 12 | 0.00 |

**Findings.**
1. The rank-quantization refusals are real end-to-end: `rrf_top` answers only
   13/26 and refuses 8 answerable questions whose relevant paper sat at hybrid
   rank 1-2; `cos_mean_top3` answers 17/26 and refuses just 1 — confirming D3.
2. The D8 trade-off is real end-to-end: `cos_mean_top3` emits answers for 2 of
   the 4 near-miss (absent-topic) queries (near abstain 0.50) where `rrf_top`
   refuses all — the reason the default stays `rrf_top`.
3. **Citation grounding is 1.00** among every emitted answer under both gates:
   each emitted versioned ID was in the retrieved set (membership, not
   entailment).
4. **Model migration:** `gpt-oss-120b` omits the `[id]` citation format more than
   Llama did, so the citation gate refuses answers as uncited (5 of 26 answerable
   for rrf_top, 8 for cos_mean_top3, 10 for the entail run; the "no-cite refusals"
   column above counts all 35 records). Correction (2026-09-07, review F2): the
   Llama v2 run had zero enforcement errors, so this is a **regression from the
   model migration, not "working as designed"** — the citation gate did its job,
   but the generation prompt needs model-specific citation tuning (follow-up,
   measured on development only).
5. **The 0.80 faithfulness gate refuses every answer** (faithfulness 0.0-0.5).
   Correction (2026-09-07, review F1): this gate is a **token-overlap heuristic
   (`HeuristicOverlapJudge`), not a trained NLI model** — no NLI model exists in
   the package; earlier "statement-level NLI gate" wording was inaccurate. So the
   honest reading is "an overlap heuristic at 0.80 refuses every gpt-oss-120b
   answer; a real NLI judge was never built." Kept as-is.

Correction (2026-09-07, review F3): the table's `refuse@rank1-2` value of 15 for
the entail row conflates causes — it is 1 gate refusal + 14 heuristic-faithfulness
refusals. `docs/results.md` now splits `gate refuse@1-2` from `heuristic
refuse@1-2`, and the RAG summary + verifier record both separately going forward.

Harness bug found and fixed: an entailment-refused record carried the discarded
answer's citations while `answer` was null; the record now emits no citations
when no answer is emitted (verifier caught it). Also added a verifier
consistency check that the gate threshold was actually applied per record, and a
`--abstain-threshold` flag to `verify-evidence` so the alternative-gate files
verify against their own frozen threshold.

**Post-run:** 60 tests pass; frozen v2 RAG evidence still verifies; all three D7
files verify. Config default still `llama-3.3` at this point (changed next
commit). D7 done; Phase 1b complete.

**Default model fix (2026-09-06, separate commit per A7).** Changed the config
default `groq_model` from the retired `llama-3.3-70b-versatile` to
`openai/gpt-oss-120b`, and corrected the stale "Groq/Llama 3.3 70B" claims in
`docs/status.md` and `PROJECT_SPEC.md`. Verified live: `papertrail ask` now
answers with citations (model `openai/gpt-oss-120b`), so `ask`/`/ask` is
functional again. Evidence for the choice: `docs/evidence/phase-8-rag-*.json`.

## 2026-09-06 — Phase 2 (Part E): Reranker evaluation

Pre-run: 60 tests passed; v2 + v3 retrieval evidence verified; DB manifest
verified (1,000 IDs, 2,039 chunks); package reinstalled (`pip install '.[dev,ml]'
--no-deps`).

Question: `hybrid_rerank` led every v3 cell, but only at one pool size with one
model. Measure the pool-size × model trade-off, its latency cost, and whether the
reranker or the fusion did the work.

Ran: `papertrail evaluate --modes <mode> --reranker <model> [--reranker-max-length
512] --rerank-pool <N>` for the eight configs
`{ms-marco, bge-reranker-base} × {pool 20, 50} × {hybrid_rerank, vector_rerank}`.
Both reranker models were confirmed to load with a printed score before the timed
runs (ms-marco logits +7.63/-11.42; bge sigmoid 0.98/0.00 on an attention-vs-
thermodynamics pair), so no model downloaded inside a latency loop (trap E).
Each config is one process (fresh model load paid in the discarded warm-up), one
mode, so its latency is clean and within-file comparable (A10).

Evidence (all verified, `--kind retrieval`, 78 rows each):
docs/evidence/phase-8-rerank-{msmarco,bge}-pool{20,50}-{hybrid_rerank,vector_rerank}.json

nDCG@10 by type (printed from the JSONs; dev is the pattern, held-out the
confirmation):

| config | d.all | d.para | d.top | h.all | h.para | h.top | dev p50 ms | dev p95 ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline hybrid (no rerank) | 0.879 | 0.890 | 0.697 | 0.913 | 0.969 | 0.683 | 57.3 | 310.5 |
| ms-marco pool20 hybrid_rerank (current) | 0.942 | 0.964 | 0.821 | 0.951 | 1.000 | 0.789 | 169.9 | 231.4 |
| ms-marco pool20 vector_rerank | 0.924 | 0.948 | 0.773 | 0.925 | 0.944 | 0.788 | 122.6 | 144.3 |
| ms-marco pool50 hybrid_rerank | 0.930 | 0.933 | 0.832* | 0.949 | 1.000 | 0.778* | 341.5 | 1419.5 |
| ms-marco pool50 vector_rerank | 0.920 | 0.933 | 0.786* | 0.926 | 0.944 | 0.793* | 226.4 | 317.5 |
| bge pool20 hybrid_rerank | 0.951 | 0.985 | 0.819 | 0.953 | 1.000 | 0.797 | 1734.4 | 5248.0 |
| bge pool20 vector_rerank | 0.940 | 1.000 | 0.741 | 0.953 | 1.000 | 0.797 | 1341.3 | 1614.3 |
| bge pool50 hybrid_rerank | 0.955 | 0.985 | 0.838* | 0.955 | 1.000 | 0.803* | 3712.8 | 11096.4 |
| bge pool50 vector_rerank | 0.942 | 0.985 | 0.778* | 0.948 | 1.000 | 0.775* | 2210.7 | 3271.0 |

Lexical nDCG@10 is 1.000 in every cell (dev and held-out) and is omitted above.
Recall@10 is 1.000 in every config on both splits: reranking reorders an already-
complete top-10 candidate set, so it moves nDCG (ranking quality), not recall.
`*` = topical lower bound: at pool 50 the first stage reaches outside the depth-20
judged pool, so some top-10 papers are unjudged and scored grade 0 (see failure
mode).

Reading:
- Reranking earns its place. Every rerank config beats plain hybrid on dev·all
  (0.879 → 0.92-0.955) and dev·topical (0.697 → 0.74-0.84); the lift is
  concentrated in topical and paraphrase, since lexical is already saturated.
- Did fusion or the reranker do the work? Both. `vector_rerank` alone already
  beats plain hybrid (dev·all 0.924 vs 0.879), so the cross-encoder does real
  work; but `hybrid_rerank` beats `vector_rerank` on dev·topical at every model
  and pool (e.g. ms-marco pool20 0.821 vs 0.773; bge pool20 0.819 vs 0.741), so
  RRF fusion contributes topical signal the reranker cannot recover from the
  vector list alone. Fusion + rerank is the best of the three.
- ms-marco vs bge: bge is marginally better on dev·all (pool20 0.951 vs 0.942)
  and on paraphrase, but costs ~10× the latency (dev p50 1734 ms vs 170 ms).
- pool 20 vs pool 50: no reliable quality gain. ms-marco pool50 is *worse* on
  dev·all (0.930 vs 0.942) and on held-out topical (0.778 vs 0.789); bge pool50
  gains only +0.004 dev·all over pool20 — and both pool-50 topical figures are
  lower bounds. pool 50 also roughly doubles latency.

Design choice: keep the current default (ms-marco-MiniLM-L-6-v2, `hybrid_rerank`,
pool 20). It captures essentially all the quality lift (+0.063 dev·all over plain
hybrid; topical 0.697 → 0.821) at ~110 ms added p50, whereas bge buys +0.009
dev·all for ~10× the latency and pool 50 buys nothing reliable for ~2× the
latency. No default changed (Part E is a measurement phase).

Failure mode found: "pool 50" is a leaky knob. `vector_rerank` never actually
reaches 50 — its pool collapses to 21-34 distinct papers because the vector
candidate list is only 100 chunks (`candidate_limit = min(200, max(50, 10*10))`),
recorded per row as `rerank_pool_size` (trap E). And `hybrid_rerank` pool 50
reaches *outside* the depth-20 judged pool on 7 dev/held-out topical rows
(ms-marco; 9 for bge), pulling 1-2 unjudged papers into the top-10 that are scored
grade 0 — so every pool-50 topical nDCG above is a lower bound. Both are recorded
in the evidence (`rerank_pool_size`, `unjudged_ranked_ids`) and the verifier
recomputes them.

Surprises / not measured: the frozen v3 baseline's `hybrid_rerank` latency (dev
p50 3896 ms, held-out p95 190 s) was polluted by model loading inside the timed
loop; the clean re-measurement here is dev p50 170 ms (~23× faster). Per A10 that
cross-run latency is not comparable — the trustworthy rerank latency is the
Phase-2 within-file number. No human has reviewed these rerank orderings; nDCG is
computed against the frozen v3 grades only.

Post-run: 64 tests passed (added 4: `vector_rerank`-skips-keyword, rerank-pool
depth, pool-below-limit guard, rerank-protocol verifier); frozen v2 and v3
retrieval evidence still verify; all 8 new files verify.

## 2026-09-07 — Review response (Phases 1, 1b, 2 findings F1–F5, S1–S4)

An independent review (Claude Fable 5.1 session, checked against the evidence
files and code) raised five must-fix items and four small ones. All addressed:

- **F4 (separate commit, code correctness):** `answer_question` computed `rrf_top`
  from `hits[0]["score"]` for any `retrieval_mode`; over `hybrid_rerank` that score
  is a cross-encoder logit, so the gate was meaningless. Latent (default mode is
  `hybrid`). Fixed to reuse the hits only when `retrieval_mode == "hybrid"` and
  otherwise recompute via `gate_signal()`; test added.
- **F1 (honesty):** the "NLI"/"entailment model" naming was false — the only judge
  is a token-overlap heuristic with no model. Renamed `HeuristicNLIJudge` →
  `HeuristicOverlapJudge` and the Protocol `NLIJudge` → `FaithfulnessJudge`; fixed
  the module docstring, the abstain reason, and every "NLI" mention in README,
  status.md, results.md, and this file. Added `entailment_judge` to the RAG
  evidence protocol for future runs (records `HeuristicOverlapJudge (token
  overlap, no model)`); the three existing D7 files predate the field, so
  results.md states the judge was the heuristic (the only one that exists).
- **F2 (honesty):** citation-format failures were under-reported. Added
  `answerable_uncited_rate` to the RAG summary and verifier, and results.md/README
  now state the uncited counts (5/8/10 of 26) beside the answer rates. Reframed the
  finding as a regression from the Llama run, not "working as designed".
- **F3 (honesty):** `answerable_refused_at_rank_1or2` conflated gate and heuristic
  refusals (the entail file's 15 = 1 gate + 14 heuristic). Split into
  `gate_refused_at_rank_1or2` and `entailment_refused_at_rank_1or2` in the summary
  and verifier (old field kept for back-compat); results.md table corrected.
- **F5 (process):** the canonical interpreter for these ML phases is **`.venv-ml`**
  (it carries the `ml` extra: sentence-transformers, the cross-encoders). The base
  `.venv` named in CLAUDE.md lacked the post-Phase-1 package and failed
  `verify-evidence` until reinstalled. Going forward the pre-run checklist prints
  `pip show papertrail-rag` (name/version/location) so a stale install is caught
  before a run, and the non-editable reinstall target is `.venv-ml`.
- Small: results.md ties the EXPLAIN "~10 ms" number to its no-setup caveat next
  to the table's 23.1 ms (S1); README/results.md state recall@50 is undefined below
  ef=50 (S2); a footnote flags the frozen v3-baseline `hybrid_rerank` latency as
  model-load-polluted and points to the clean Phase 2 figure without editing the
  frozen file (S3); results.md notes `ce_margin` would have won under a near-miss-
  cost rule, a limitation of AUROC-only selection (S4).

The existing frozen evidence (v2 retrieval/RAG, v3 baseline, hnsw, gate, the three
D7 RAG files, the eight rerank files) all still verify unchanged — the new RAG
summary fields are optional in the verifier. No default and no evidence file was
changed; no question set was touched.

Follow-up opened: a model-specific citation prompt for `gpt-oss-120b`, to be tuned
and measured on development questions only before any further RAG evidence.

## 2026-09-08 — Phase 3 (Part F): Sparse retrieval

Pre-run: 66 tests passed; DB up, manifest verified (1,000 IDs, sha 7308d240…);
package reinstalled into `.venv-ml` — `papertrail-rag 0.1.0`, site-packages in
`.venv-ml/lib/python3.12` (F5 check); frozen v2 (90 rows) and v3 baseline (312
rows) both verify.

Why this phase: keyword is the weakest retriever in the v3 baseline — dev nDCG@10
0.759 overall, 0.695 on paraphrase, and 0.497 on the 6 held-out topical questions.
Today it is OR-of-all-terms ranked by `ts_rank_cd`, with no length normalization
and no field weighting.

**Decision rule, pre-registered before running anything (brief Part F):**
> If offline BM25 beats Postgres FTS by **≥ 0.05 nDCG@10** on development
> `lexical` **or** `paraphrase`, the *ranking function* is what matters, and I go
> to F2-iii (field weights + length normalization). If the gap is **< 0.05**, the
> *query construction* matters more, and I do F2-i (AND-then-OR cascade) first.

Recording this before I look at any BM25 number so the branch is not chosen after
the fact. Known confound to state up front, not to "fix": Postgres FTS stems
(`english` config) and the offline BM25 does not tokenize-and-stem the same way,
so this is a ranking-function comparison under different tokenization, not a
controlled single-variable swap (brief trap).

**F1 — offline BM25 ablation.** `eval/tools/bm25_ablation.py` (read-only, no DB
writes). `rank_bm25 0.2.2`, `BM25Okapi(k1=1.5, b=0.75, epsilon=0.25)` over all
2,039 chunk texts, tokenized with exactly `keyword_search`'s regex
(`[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*`) lower-cased; query terms under 3 characters
dropped, as `keyword_search` does. Top-200 chunks → `distinct_papers` → 10.
Evidence: `docs/evidence/phase-8-bm25-offline.json` (verified, `--kind bm25`,
52 rows).

**Development-only, deliberately.** Held-out is reserved for the one final report
on the best development configuration (F3 / A1), so this diagnostic never touched
it. The verifier enforces it: a held-out row in this file is a hard error.

nDCG@10, development, versus the frozen v3 baseline's `keyword` rows:

| type | FTS-OR (`ts_rank_cd`) | BM25 offline | gap |
|---|--:|--:|--:|
| all | 0.759 | 0.887 | +0.128 |
| lexical | 0.977 | 1.000 | +0.023 |
| paraphrase | 0.695 | 0.917 | **+0.222** |
| topical | 0.595 | 0.675* | +0.080* |

Recall@10 moves too: all 0.923 → 0.981, paraphrase 0.833 → 0.958.
`*` topical is a **lower bound**: BM25 was not one of the four retrievers that
built the frozen topical pools, so it surfaces unjudged papers — all 12 topical
rows do, 2–7 unjudged ids each in the top 10 — and those score grade 0. Do not
read the topical row as a clean comparison.

**Decision rule fires on paraphrase (+0.222 ≫ 0.05): the ranking function
matters → F2-iii next.** The rule was pre-registered above before any BM25 number
existed, and it happens to read the two types (lexical, paraphrase) whose
relevance is a fixed known-item set with no pool, so the pooling bias above cannot
have influenced the branch.

**Why FTS loses, confirmed rather than assumed.** On the two worst paraphrase
questions the gap is total — BM25 nDCG 1.000 vs FTS 0.000:

| question | relevant paper | BM25 paper-rank | FTS paper-rank | FTS matched set |
|---|---|--:|--:|--:|
| v3q026 | 2608.03291v1 | 1 | **14** | 978 chunks / 639 papers |
| v3q004 | 2608.01085v1 | 1 | **14** | 1,486 chunks / 880 papers |

FTS **did** match the right paper in both cases and simply ranked it 14th, just
outside the top-10 cutoff. So this is a **ranking failure, not a matching
failure** — which is exactly what the decision rule's branch claims, and it is why
the fix belongs in the rank expression rather than in the query builder. Mechanism:
`ts_rank_cd` scores from within-document term frequency and cover density only —
it has no corpus-wide IDF term — and at the default normalization flag it does not
divide by document length. On a 21–25-term natural-language paraphrase, common
words then contribute as much as the rare discriminative ones ("satisfiability",
"dormant"). BM25 weights by IDF and normalizes by length (`b=0.75`). F2-iii adds
both of the missing pieces on the Postgres side: `setweight` field weights and
`ts_rank_cd(..., 32)`, whose flag divides by document length.

Design choice: kept BM25 unstemmed rather than matching the Postgres `english`
stemmer. Matching stemmers would be a different study; leaving it means this is a
ranking-function comparison *under different tokenization*, stated in the evidence
file's claim boundary rather than papered over.

Not measured: BM25 as a *served* retriever. This is an in-process index built in
~0.6 s over 2k chunks; its per-query time (dev p50 2.9 ms) is in-process scoring
and is **not** comparable to the SQL path (A10). Whether ParadeDB/`pg_search`
would be worth it is F2-iv, and only if F2-iii fails to close the gap.

**F2-iii — field weights + length normalization. Negative result: it makes things
worse.** Evidence: `docs/evidence/phase-8-keyword-{or,or-depth200,weighted-n0,
weighted-n1,weighted-n2}.json` (all verified, `--kind sparse`, 52 dev rows each).

Two corrections to the brief's spec, both confirmed before building (Mayank chose
the corrected plan):

1. **`ts_rank_cd(..., 32)` is not length normalization.** The brief says flag 32
   "divides by document length"; in Postgres, flag 32 divides the rank by itself
   plus one, 2 divides by length, 1 by 1+log(length). Because x/(x+1) is
   monotonic, flag 32 **cannot reorder anything**. Checked directly on v3q026:
   flags 0 and 32 return an identical top-15 ordering (top score 1.300 vs 0.565,
   the same ranking rescaled); flags 1, 2 and 16 do reorder. So the literal spec
   would have measured field weights only, with zero normalization effect.
2. **Additive migration, not a replacing one.** `20260908_0003` *adds*
   `search_vector_weighted` (title `A`, body `B`) plus its GIN index and leaves
   `search_vector` untouched. Replacing it would have silently reordered the
   current `"or"` default too (all-D lexemes become A/B under the default rank
   weights), breaking reproducibility of the frozen keyword rows and making
   "cascade" vs "cascade+weights" unmeasurable. `alembic downgrade -1` then
   `upgrade head` both proven; `\d chunks` shows both GIN indexes.

nDCG@10, development:

| variant | all | lexical | paraphrase | topical |
|---|--:|--:|--:|--:|
| `or` (frozen baseline) | 0.759 | 0.977 | 0.695 | 0.595 |
| `or-depth200` | 0.759 | 0.977 | 0.695 | 0.595 |
| `weighted-n0` (weights only) | 0.721 | **1.000** | 0.602 | 0.588 |
| `weighted-n1` (÷1+log len) | 0.729 | **1.000** | 0.623 | 0.580 |
| `weighted-n2` (÷ len) | 0.480 | 0.923 | 0.314 | 0.222 |
| BM25 offline (F1) | 0.887 | 1.000 | 0.917 | 0.675 |

Two things this table settles before the main result:
- **`or` reproduces the frozen v3 baseline exactly** (0.759 / 0.977 / 0.695 /
  0.595). The additive migration left the default bit-identical, and the harness
  is validated against frozen evidence.
- **`or-depth200` is identical to `or`**, so the candidate-depth asymmetry in F1
  (BM25 read 200 candidate chunks, the SQL path 100) is worth exactly **zero**
  nDCG. The F1 comparison was fair; that confound is now closed rather than
  hand-waved.

**The result: field weighting helps `lexical` and hurts `paraphrase`, and the net
is negative** (all 0.759 → 0.721). Length normalization does not rescue it (n1
0.729) and full division by length is catastrophic (n2 0.480).

Why, measured rather than guessed: v3 paraphrase queries are constructed so no
title word appears in the query. On the development split the relevant paper's
title shares a mean of **0.04** content words with its paraphrase query — **23 of
24 share none at all** — versus **4.81** for lexical queries, where none has zero
overlap. So `setweight(title, 'A')` boosts, at weight 1.0, exactly the field a
paraphrase query cannot match, and dilutes the body evidence (weight 0.4) that it
can. Lexical rises to a perfect 1.000 for the same reason, in reverse. The knob is
not broken; it is aimed at the wrong type.

**Conclusion: the missing ingredient is IDF, and Postgres's rank knobs cannot
supply it.** `ts_rank_cd` scores from within-document term frequency and cover
density; it has no corpus-wide document-frequency term at all. Field weighting and
length normalization are the two levers Postgres offers, and on this question set
both are neutral-to-harmful. That is exactly the brief's precondition for F2-iv
(ParadeDB `pg_search`, which implements real BM25): F1 showed the ranking function
matters by a wide margin (+0.222) *and* iii did not close the gap.

No default changed: `keyword_search` still ranks the unweighted column with the
original expression unless `weighted=`/`normalization=` are passed (A7).

**F2-i — AND-then-OR cascade.** `keyword_search(strategy="cascade")`: AND of every
term first (each term quoted so a hyphenated token is a lexeme, not an operator),
then fill the remainder from the OR query with AND rows kept in front. Default
stays `"or"` (A7). Evidence: `docs/evidence/phase-8-keyword-cascade.json`.

Development: all 0.759 → **0.766**, lexical 0.977 → **1.000**, paraphrase 0.695 →
0.695, topical 0.595 → 0.595. The gain is lexical-only, and the mechanism is
measured: the AND branch returns at least one row on **8 of 16** lexical questions
(mean 8.9 query terms) and on **0 of 24** paraphrase and **0 of 12** topical ones
(mean 23.7 and 10.5 terms). ANDing two dozen terms matches nothing, so on 36 of 52
development questions the cascade is a provable no-op that falls through to the
identical OR query.

Both brief traps checked directly against Postgres rather than assumed, and both
turned out milder than written: an all-stop-word AND (`'the' & 'and' & 'for'`)
yields an **empty tsquery, not an error**, which matches nothing and falls through
to the OR fill; and a hyphenated token parses as a lexeme with a phrase expansion
(`'state-of-the-art' <-> 'state' <3> 'art'`) whether or not it is quoted, so the
hyphen is never read as NOT. The terms are quoted anyway, and both cases are
covered by tests (`tests/test_keyword_strategies.py`) so a future Postgres that is
stricter fails loudly.

**F2-ii — phrase boost. Negative.** `strategy="cascade_phrase"` ranks
`phraseto_tsquery` matches (quoted spans, then adjacent capitalised word pairs)
ahead of the cascade. Applicability is small by construction: only **3 of 52**
development queries contain a quoted span and **4** contain an adjacent capitalised
pair. Development: all 0.766 → **0.746**, lexical 1.000 → **0.935**; paraphrase and
topical unchanged. Forcing phrase hits to the front promotes chunks that contain the
bigram but are worse overall, displacing better AND/OR matches. The brief guessed
this "may do nothing"; measured, it does something small and harmful.

**F3 — report and recommendation.** Development, every row measured by the same
harness (`eval/tools/keyword_variants.py`, 52 questions), nDCG@10:

| variant | all | lexical | paraphrase | topical |
|---|--:|--:|--:|--:|
| FTS-OR (current default) | 0.759 | 0.977 | 0.695 | 0.595 |
| `or-depth200` (depth control) | 0.759 | 0.977 | 0.695 | 0.595 |
| cascade (F2-i) | **0.766** | 1.000 | 0.695 | 0.595 |
| cascade + phrase (F2-ii) | 0.746 | 0.935 | 0.695 | 0.595 |
| cascade + weights (F2-iii) | 0.721 | 1.000 | 0.602 | 0.588 |
| weights only, n0 | 0.721 | 1.000 | 0.602 | 0.588 |
| weights, n1 (÷1+log len) | 0.729 | 1.000 | 0.623 | 0.580 |
| weights, n2 (÷ len) | 0.480 | 0.923 | 0.314 | 0.222 |
| **BM25 offline (F1)** | **0.887** | 1.000 | **0.917** | 0.675* |

Held-out, run **once** on the best development configuration (cascade), against
the frozen baseline's held-out keyword row — no extra held-out read was needed for
the baseline because it already exists in the frozen file:

| held-out | all | lexical | paraphrase | topical |
|---|--:|--:|--:|--:|
| FTS-OR (frozen) | 0.762 | 1.000 | 0.735 | 0.497 |
| cascade | 0.762 | 1.000 | 0.735 | 0.497 |
| gap | +0.000 | +0.000 | +0.000 | +0.000 |

**The cascade's development gain did not replicate: it is exactly zero on
held-out, on every type.** The reason is visible rather than mysterious — the whole
dev gain was lexical 0.977 → 1.000, and held-out lexical was *already* 1.000, so
there was nothing left to win. A sub-question-sized improvement on a near-saturated
type is not a result.

**Recommendation to Phase 4: change nothing in the keyword path.** Every
Postgres-side lever was measured and none earns adoption — cascade is +0.007 on
development and +0.000 on held-out, the phrase boost is negative, and field
weighting and length normalization are negative. The one large, reproducible signal
is BM25's **+0.222** on development paraphrase, and its cause is IDF, which
`ts_rank_cd` structurally does not have and which none of Postgres's rank knobs can
supply. Closing it means a different ranking engine (F2-iv, ParadeDB `pg_search`) —
a second Compose service and explicitly not part of the default stack — which is a
Phase 4 decision for Mayank, not something to adopt inside a measurement phase.

Worth carrying into Phase 4: keyword is only one of two RRF inputs, and the
vector+rerank path already handles paraphrase well (dev `hybrid_rerank` paraphrase
0.964 vs keyword 0.695), so keyword's paraphrase weakness is partly absorbed by
fusion. Whether it should be down-weighted rather than repaired is exactly the
Part G question.

Post-run: 75 tests pass (8 new keyword-strategy tests); every prior evidence file
still verifies (v2, v3 baseline, 8 rerank, hnsw, gate, 10 sparse); manifest intact.

## 2026-09-09 — Phase 4 (Part G): Fusion ablation

Pre-run: 75 tests passed; DB up, manifest verified (1,000 IDs, sha 7308d240…);
package reinstalled into `.venv-ml` — `papertrail-rag 0.1.0` (F5 check); frozen v2
and v3 baseline both verify; tree clean at `bca31fc`.

Why this phase: RRF `k=60`, equal weights, and the 50–200 candidate pool were
inherited, never ablated. Phase 3 also left an open question — keyword is the
weakest retriever (dev nDCG 0.759 vs `hybrid_rerank` 0.942) and no Postgres fix
helped, so the live question is whether fusion should *down-weight* keyword rather
than try to repair it. Phase 1 additionally flagged that the default
`hnsw.ef_search=40` truncates the vector candidate list below `candidate_limit`
(50–200), which this phase must avoid by setting `ef_search` explicitly.

**Selection rule, pre-registered before the sweep runs (brief G3):**
> Choose the configuration with the highest **development** nDCG@10 on `all`.
> Ties broken by `paraphrase`, then `lexical`. Among configurations still tied,
> prefer the **simplest**: unweighted RRF over weighted, and the smaller candidate
> pool over the larger.

Recorded now, before any sweep number exists, so the winner cannot be chosen after
the fact. Two things I am binding myself to up front (brief G traps): the sweep is
**development only** and is *not* a reportable result — only the G4 held-out run
is; and no default changes in this phase without Mayank seeing G4 first (A18/G6).

**G1–G2 — parameterized fusion and the sweep.** `reciprocal_rank_fusion` gained
`weights` (each source contributes `w/(k+rank)`; `weights=None` is the original
function, and the existing RRF tests pass unchanged). Added `convex_fusion`:
min-max normalize each source's own scores into [0,1], then
`alpha*vector + (1-alpha)*keyword`, missing source contributes 0, same
deterministic tie-break. `retrieve()` gained `weights`, `fusion`, `alpha`,
`candidate_limit`, `ef_search`, `keyword_strategy`.

Sweep: `k ∈ {10,30,60,100}` × `w_vec ∈ {1,2,3}` plus `alpha ∈ {0.5,0.7,0.9}`, each
× `candidate_limit ∈ {50,200}` × keyword `{or, cascade}` = **60 configurations ×
52 development questions = 3,120 rows**, in 26 s. Evidence:
`docs/evidence/phase-8-fusion-sweep-dev.json` (verified, `--kind fusion`).

Design choice worth defending: for each question the two candidate lists are
fetched **once** per (pool, keyword strategy) and every configuration is fused from
those same cached lists. So a difference between cells is caused by the fusion and
nothing else. The cost is that **per-config latency is not measured**, and rather
than emit a fabricated `0.0` I made `_aggregate` omit latency keys when no row
carries a measurement (A5). Candidate-fetch latency is recorded per
(pool, strategy) instead: cl50 p50 44 ms, cl200 p50 69 ms.

**What actually moves the number — k, not the pool:**

| k (unweighted RRF, OR) | dev all @ cl50 | @ cl200 |
|---|--:|--:|
| 10 | 0.892 | **0.900** |
| 30 | 0.875 | 0.876 |
| 60 (current) | 0.875 | 0.877 |
| 100 | 0.875 | 0.877 |

Widening the candidate pool 50 → 200 is worth +0.008 at k=10 and +0.001 elsewhere.
Nearly the whole development gain is **k**. Mechanism: RRF's discount is
`1/(k+rank)`, and at k=60 that is almost flat across the top ten — 1/61 vs 1/70 is
a 13% spread — so fusion barely distinguishes rank 1 from rank 10 and the weaker
keyword list drags good vector hits down. At k=10 the spread is 1/11 vs 1/20, 45%,
so the top of each list dominates. Consistent with that reading, up-weighting the
vector side at k=60 buys most of the same thing by another route (w_vec 1 → 3:
0.877 → 0.895), and once k=10 the weighting is unnecessary.

**G3 — selection.** The pre-registered rule picked **`rrf-k10-wv1-cl200-or`**
(RRF k=10, unweighted, pool 200, ef_search 200, OR keyword). Honest note on the
tie-break: the top two cells — the `or` and `cascade` keyword variants — are tied
to the last floating-point digit on every type (0.9001742652741194 on `all`), even
though their ranked lists differ on 3 of 52 questions. My first sort broke that tie
by alphabetical config id, which picked `cascade` for no reason at all. I extended
the "simplest" clause of the rule to prefer the incumbent `or`, put it in
`select_fusion_config` so the sweep tool and the verifier use one implementation,
and the verifier now recomputes it.

**G4 — held-out, once, chosen vs the v2-style baseline** (k=60, equal weights,
pool 50, `ef_search` at the Postgres default, OR keyword). Evidence:
`docs/evidence/phase-8-fusion-heldout.json` (verified, cross-checked against the
sweep).

| held-out (n=26) | baseline | chosen | gap |
|---|--:|--:|--:|
| all | 0.916 | 0.921 | +0.005 |
| paraphrase (12) | 0.969 | 0.969 | +0.000 |
| lexical (8) | 1.000 | 1.000 | +0.000 |
| topical (6) | 0.698 | 0.721 | +0.023 |

Recall@10 is 1.000 for both on every type; latency p50 60.3 ms chosen vs 61.7 ms
baseline (within-file only, A10).

**The development gain does not transfer.** +0.025 on development becomes **+0.005
on held-out**, and the only cell that moves at all is topical — **6 questions**
(A11). Paraphrase and lexical are flat to three decimals. I am not going to call a
+0.005 overall change on 26 questions a win.

**A prediction of mine that the data killed.** I set the baseline's `ef_search`
unset expecting it to truncate the vector list to 40 — the Phase 1 trap — and wrote
that into the file's protocol note. The per-row `vector_candidates` came back
**50, not 40**. `EXPLAIN (ANALYZE)` explains why: at LIMIT 50 *and* LIMIT 200 the
planner runs an exact **Seq Scan** rather than the HNSW index, so the ef cap never
binds. `SHOW hnsw.ef_search` is indeed 40; it simply does not apply when the index
is not used. So at 2,039 vectors the Phase 1 truncation concern **does not bite in
production**. I corrected the false note and re-ran before committing — the run is
deterministic and every metric was byte-identical, and no committed file ever
carried the wrong claim.

**G5 — verifier.** `verify_fusion_evidence` (`--kind fusion`) recomputes every row
and every per-config aggregate, enforces sweep = development-only / held-out =
heldout-only, checks RRF k against the report's **own** declared `rrf_k_values`
rather than the frozen constant 60, and — given `--sweep` — recomputes the
selection rule over the sweep and rejects a held-out file that names anything other
than the development-best configuration. Tested both ways.

**G6 — no default changed.** Stopping here for Mayank to see the G4 table (A18).

Post-run: 83 tests pass; every prior evidence file still verifies (v2, v3 baseline,
8 rerank, 10 sparse, hnsw, gate); manifest intact.

### Correction (2026-09-10, review F1–F4): the Phase 4 ef_search conclusion was wrong

An independent review found that my "the ef cap never binds in production" finding
was an artifact of **leaked Postgres session state**, and it is right. I reproduced
it before changing anything:

| call (one transaction) | rows |
|---|--:|
| `vector_search(limit=50)` first in a fresh transaction | **40** |
| `vector_search(limit=50, ef_search=200)` | 50 |
| `vector_search(limit=50)` again, same transaction | **50** ← inherited |
| `vector_search(limit=200)` again, same transaction | **200** ← inherited |
| `vector_search(limit=50)` in a new transaction | **40** |
| `vector_search(limit=200)` in a new transaction | **40** |

`SET LOCAL` lasts for the rest of the transaction, and `vector_search` only issued
one when `exact=True` or `ef_search` was given — so a call passing neither inherited
whatever came before. `fusion_heldout.py` ran the chosen configuration
(`ef_search=200`) and then the baseline (unset) inside one `session_scope`, so the
baseline silently ran at ef=200.

**The Phase 1 truncation finding was correct all along.** With `ef_search` unset the
production query uses the index and returns **40** candidates regardless of
`candidate_limit` — including at `candidate_limit=200`. My earlier `EXPLAIN` "proof"
misled me because I wrote a different query form (`WHERE embedding IS NOT NULL` with
a raw `<=>`), not the production one, and that form planned as a Seq Scan.

Fixes, in order:
- **F2** (`28fd055`): `vector_search` now sets **every** GUC it touches on every
  call, to a value or to `DEFAULT`; validation moved ahead of the first `SET LOCAL`.
  New skipped-without-a-DB integration tests pin all three leaks. Added **A12b** to
  the brief.
- **F3**: G4 re-run as three configurations, each in **its own transaction**, into
  `docs/evidence/phase-8-fusion-heldout-v2.json`. The old file is kept unedited and
  marked superseded (A3). The verifier now asserts a configuration's declared
  `expected_vector_candidates` against every row, which is exactly what catches a
  leak.
- **F4**: the false prose is reverted in `docs/results.md` (both the Phase 1 and
  Phase 4 sections) and here. Commit `cfdd6b7` was wrong.

Corrected held-out numbers:

| held-out (26) | production as-is | production, ef fixed | chosen |
|---|--:|--:|--:|
| vector candidates | **40** | 50 | 200 |
| all | 0.913 | 0.916 | 0.921 |
| paraphrase (12) | 0.969 | 0.969 | 0.969 |
| lexical (8) | 1.000 | 1.000 | 1.000 |
| topical (6) | 0.683 | 0.699 | 0.721 |
| latency p50 ms | 34.8 | 38.6 | 39.5 |

The strongest check that this run is measuring the real system: `production as-is`
reproduces the **frozen v3 baseline's held-out hybrid nDCG of 0.913 exactly**. The
leaked run had read 0.916.

Decomposition of the +0.009: **lifting the ef cap alone is worth +0.003 on `all`
and +0.016 on topical**; k=10 plus the wider pool adds the remaining +0.005 /
+0.022. Every cell that moves is topical — 6 questions.

Not over-read: `fusion_sweep.py` passes `ef_search` explicitly on **every** call, so
the development sweep was never contaminated. Its `cl50` cells ran at ef=50 and its
`cl200` cells at ef=200, which means no sweep cell is the production configuration.
`hnsw_study.py` was accidentally safe because it opens a session per call.

Lesson worth keeping: a helper that mutates session state must restore it, and a
harness that runs configurations back to back in one transaction will silently
compare a configuration against itself. The tell was available and I missed it —
`vector_candidates` was recorded per row all along, and 50 in the baseline should
have contradicted a documented cap of 40.

### G6 decision (2026-09-12): adopt the ef_search fix, keep RRF k=60

Mayank approved the reviewer's recommendation (a) and declined (b).

**(a) Adopted — `retrieve()` defaults `ef_search` to `max(candidate_limit, 40)`.**
This is a correctness fix, not a tuning choice: the function asked for 50–200 vector
candidates and silently got 40. Live effect at the default `limit=10`: 40 → 100
candidates. Evidence: `docs/evidence/phase-8-fusion-heldout-v2.json`
(production as-is 0.913 vs production with the cap lifted 0.916 on held-out `all`;
+0.016 on topical).

Check I ran before adopting, because the abstention threshold was selected under the
40-candidate regime and nobody had asked whether more candidates move `rrf_top`:

| development | rrf_top changed | gate decision flipped |
|---|--:|--:|
| positives (52) | 1 | **0** |
| negatives (18) | 6 | **0** |

Zero flips on all 70, so the frozen threshold `0.03239446668849102` remains valid
and the gate study does not need redoing. Top-10 ordering changes on 41 of 52
development questions — that is the fix working, not a side effect.

Consequence to state plainly: the frozen v3 baseline, the Phase 2 rerank files and
the v2 evidence were all produced under the 40-candidate regime. They still verify
(verification recomputes from stored rows) and they remain the honest record of that
configuration, but a fresh run of `evaluate` will no longer reproduce them
byte-for-byte. That is the expected cost of fixing the cap.

**(b) Declined — RRF `k` stays 60.** The sweep's k=10 winner gains +0.009 on held-out
`all` over production, but +0.003 of that is the ef fix above and every remaining
cell that moves is topical, on 6 questions. Not evidence at that n (A11).

## 2026-09-13 — Phase 5 (Part H): Embedding model ablation

Pre-run: 86 tests passed (83 + the 3 DB-only session-state tests, which skip when
no database is up); DB restarted — Docker Desktop had stopped between sessions
again (A9) — manifest verified (1,000 IDs, sha 7308d240…); package reinstalled into
`.venv-ml` (`papertrail-rag 0.1.0`); frozen v3 baseline verifies (312 rows). The
database is 30 MB with one complete embedding run (MiniLM, 384-d, 2,039 chunks), so
there is room for per-model columns at ~3 MB each.

**H2 — prefixes checked against the model cards, not the brief's table.** The brief
says to verify; the table turned out to be correct, and here are the primary-source
lines.

- `all-MiniLM-L6-v2` (384-d, max_seq 256): card documents no prefix. None/none.
- `BAAI/bge-small-en-v1.5` and `BAAI/bge-base-en-v1.5` (384-d / **768-d**, max_seq
  512): the Model List table gives the query instruction as
  `Represent this sentence for searching relevant passages: `, and the card is
  explicit about the asymmetry — "If you need to search the relevant passages to a
  query, we suggest to add the instruction to the query; in other cases, no
  instruction is needed… In all cases, **no instruction** needs to be added to
  passages." So query prefix = that instruction, passage prefix = none.
- `intfloat/e5-small-v2` (384-d, max_seq 512): "Each input text should start with
  "query: " or "passage: "." → `query: ` / `passage: `.
- `thenlper/gte-small` (384-d, max_seq 512): no prefix mentioned anywhere in the
  card. None/none.

Two things the cards say that the brief's table cannot:
1. **bge v1.5 was specifically trained to work without the instruction** — the card's
   release note says v1.5 "enhance[s] its retrieval ability **without**
   instruction", and frames the instruction as for "s2p (short query to long
   passage)" retrieval. Our paraphrase queries average ~24 content words, which is
   not a short query. So the instruction may buy little here. Since bge's passage
   prefix is empty, the indexed vectors are prefix-free and I can test this from the
   **same column** by changing only the query prefix — a free ablation, added as
   `bge-small-noinstruct`.
2. `SentenceTransformer.prompts` reports `{'query': '', 'document': ''}` for **every**
   one of these models, i.e. empty defaults from sentence-transformers 5.x rather
   than model-specific prompts. So the library will not apply any prefix for us;
   prefixes must be applied explicitly by our own code. Worth knowing — relying on
   `model.prompts` would have silently produced the no-prefix case everywhere.

Storage decision: **per-model columns**, the brief's preferred option. The frozen
`embedding` column is never touched, which makes the H4 byte-identical check
structural rather than a matter of re-embedding determinism.

Controls planned, beyond the four models:
- `e5-small-noprefix` — its own column, no prefix at index or query time (H3.4 as
  written).
- `e5-small-query-prefix-missing` — the **main** e5 column (indexed with
  `passage: `) queried with no prefix. This is precisely the trap the brief names,
  and it costs nothing since it reuses the correct column.

**H3 — five columns embedded, 2,039/2,039 each.** Wall times and index sizes:

| model | dims | embed wall | encode 256 chunks | chunks/s | HNSW index |
|---|--:|--:|--:|--:|--:|
| all-MiniLM-L6-v2 | 384 | not measured here¹ | 0.90 s | 283 | 4,088 kB |
| bge-small-en-v1.5 | 384 | 58.7 s | 2.01 s | 127 | 4,088 kB |
| gte-small | 384 | 83.8 s | 2.04 s | 126 | 4,088 kB |
| e5-small-v2 | 384 | 119.0 s | 3.21 s | 80 | 4,088 kB |
| e5-small-v2 (no prefix) | 384 | 117.0 s | — | — | 4,088 kB |
| bge-base-en-v1.5 | **768** | 219.8 s | 5.71 s | 45 | **8,168 kB** |

¹ MiniLM's column was filled in an earlier phase on another day, so its embed wall
time is not comparable (A10) and is not restated. The encode column is the
comparable measure: one run, same 256 chunks, warm-up discarded. Database grew
30 MB → 85 MB for five extra columns and their indexes.

> **Superseded 2026-09-15.** The wall times and throughput above were read off the
> terminal and never written to evidence (an A4 violation found in the Phase 4–5
> audit). They are replaced by `docs/evidence/phase-8-embed-costs.json` — see that
> day's entry. The index sizes were confirmed exactly.

**Development nDCG@10** (52 questions, `--splits development` so held-out stayed
reserved; `*` marks a pooling-biased lower bound, see below):

| config | vector all | v. para | v. lex | v. topical | hybrid all | hybrid_rerank all |
|---|--:|--:|--:|--:|--:|--:|
| MiniLM (incumbent) | 0.876 | 0.883 | 1.000 | 0.699 | 0.877 | 0.940 |
| **bge-base (768-d)** | **0.918** | **0.935** | 1.000 | 0.775* | **0.910** | 0.940 |
| gte-small | 0.880 | 0.832 | 1.000 | 0.816* | 0.898 | 0.934 |
| bge-small | 0.860 | 0.802 | 1.000 | 0.788* | 0.900 | 0.935 |
| bge-small, no instruction | 0.855 | 0.797 | 1.000 | 0.777* | 0.887 | 0.939 |
| e5-small (correct prefixes) | 0.852 | 0.823 | 1.000 | 0.714* | 0.877 | 0.913 |
| e5-small, no prefix anywhere | 0.833 | 0.795 | 1.000 | 0.687* | 0.879 | 0.937 |
| e5-small, **query prefix forgotten** | 0.823 | 0.767 | 1.000 | 0.700* | 0.873 | 0.942 |

**The pools are MiniLM-specific, and this is the phase that exposed it.** Topical
relevance is judged only inside a frozen pool built from MiniLM-based retrievers, so
a different encoder surfaces papers nobody judged, scored grade 0:

| config | topical rows with unjudged ids | mean unjudged per top-10 |
|---|--:|--:|
| MiniLM | 4 / 36 | 0.22 |
| bge-small | 22 / 36 | 1.19 |
| bge-base | 26 / 36 | 1.44 |
| gte-small | 27 / 36 | 1.39 |
| e5-small | 28 / 36 | 1.64 |

So every starred topical figure is a **lower bound**, and the bias is much larger for
the new encoders than for MiniLM. Two consequences, stated rather than glossed: the
topical column cannot be used to rank the new encoders against *each other*; but
because the bias runs *against* them, "gte-small (0.816) and bge-base (0.775) beat
MiniLM (0.699) on topical" is still a valid one-directional conclusion. Paraphrase
and lexical have no pool — relevance there is a fixed known-item set — so those
columns are unbiased and are where the headline lives. The verifier now requires every
out-of-pool id to be recorded per row on every mode, which is what turned this from an
invisible understatement into a measured one (it first showed up as 8 of 10 files
failing verification).

**The prefix controls did their job.** On development vector `all`:
correct e5 **0.852** > no prefix anywhere **0.833** > **query prefix forgotten 0.823**.
Indexing with `passage: ` and then querying bare is worse than never using prefixes
at all — the mismatch costs 0.029 against correct usage, and 0.010 against simply not
bothering. That is the brief's named trap, measured. It also justifies the harness
choice: `--query-prefix` now defaults to the prefix recorded in the column's
embedding run, so this failure cannot happen by omission; producing it required
passing `--query-prefix ""` deliberately.

**bge's instruction is nearly free to omit**: 0.860 with it vs 0.855 without
(+0.005). That matches the model card, which says v1.5 "enhance[s] its retrieval
ability without instruction" and frames the instruction as for short queries — ours
average ~24 content words.

**Held-out, one read per configuration** (`phase-8-embed-{minilm,bge-base}-heldout.json`):

| held-out (26) | MiniLM | bge-base | gap |
|---|--:|--:|--:|
| vector all | 0.894 | 0.912 | **+0.018** |
| vector paraphrase (12) | 0.883 | 0.912 | +0.029 |
| vector topical (6) | 0.776 | 0.795* | +0.019 |
| **hybrid all** | 0.916 | 0.915 | **−0.001** |
| hybrid paraphrase | 0.969 | 0.928 | −0.042 |
| hybrid topical | 0.697 | 0.775* | +0.078 |
| hybrid_rerank all | 0.943 | 0.949 | +0.006 |
| vector p50 / p95 ms | 26.7 / 50.5 | 51.3 / 85.0 | ~2× |

Recall@10 is 1.000 for both on every type.

**Finding: the encoder upgrade is real and the pipeline absorbs it.** bge-base is
clearly the better encoder in isolation — +0.042 development and +0.018 held-out on
`vector`, +0.052 / +0.029 on paraphrase. In the configuration PaperTrail actually
serves (`hybrid`), the held-out difference is **−0.001**; under `hybrid_rerank` it is
+0.006. Per the brief's trap note, changing the encoder changes only the candidate
list the cross-encoder sees, and indeed every model converges to 0.913–0.942 under
`hybrid_rerank` — including the deliberately broken prefix configuration at 0.942,
which is the sharpest illustration that reranking masks encoder quality.

**Recommendation: do not change the default.** bge-base costs 3.7× the embed wall
time, 6.3× the per-chunk encode time, 2× the index size and ~2× the query-time vector
latency, and returns −0.001 on held-out `hybrid`. The case for switching would be a
product that serves pure vector search, which this one does not.

**Decision (2026-09-13, Mayank): keep `all-MiniLM-L6-v2`.** No default changed. The
alternative columns and their runs stay in the database and the evidence stays on
record, so the comparison is reproducible and the decision is revisitable if the
serving path ever changes — in particular, if PaperTrail ever exposed pure vector
search, `bge-base-en-v1.5` is the measured pick.

Worth naming as a cross-phase pattern, since it is now four for four: Phase 2 (a
better reranker), Phase 3 (a better sparse ranker), Phase 4 (better fusion
parameters) and Phase 5 (a better encoder) each produced a real component-level
improvement that shrank to roughly nothing once measured end-to-end on held-out.
The one change that did survive was a **bug fix** — the `ef_search` cap.

**H4 — the frozen baseline is byte-identical.** `phase-8-retrieval-v3-baseline.json`
still verifies, with 0 out-of-pool rows, and MiniLM's `embedding` column still holds
2,039 vectors. Because the migration was additive, this is structural: no
re-embedding of that column ever happened, so there is nothing for non-determinism
to break.

Post-run: 93 tests pass; all 10 new files verify plus every prior file (v2, v3
baseline, 8 rerank, 10 sparse, 3 fusion, hnsw, gate, v2 RAG); manifest intact.
