# Review of Phases 3 and 4 (commits 05f6f3e..cfdd6b7) — 2026-09-10

Reviewer: Claude Fable 5.1 session, for Mayank Ghadia. Checked by re-running
verifiers and tests in both interpreters and by reproducing the disputed
database behaviour directly, not by reading the notes.

## Verified OK

- All 12 Phase 3 files and both Phase 4 files verify; frozen v2, v3 baseline,
  hnsw, gate, rerank, and RAG files still verify. 83 tests pass in `.venv-ml`
  and `.venv`.
- Phase 3 is clean: the decision rule was pre-registered, BM25 was
  development-only and the verifier enforces it, the additive migration left
  the `or` default byte-identical (the `or` variant reproduces the frozen
  baseline exactly), the flag-32 correction is right, every negative result is
  kept, and no default changed. The recommendation to leave the keyword path
  alone is the correct reading.
- Phase 4 sweep: development-only, selection rule pre-registered, tie-break
  fixed and moved into shared code, latency correctly omitted instead of
  fabricated. All good.

## One finding that changes the Phase 4 conclusion

**F1. The "ef cap never binds in production" claim is an artifact of leaked
session state. The Phase 1 truncation finding was correct.**

Reproduced today on the live database, one transaction:

| call | rows returned |
|---|--:|
| `vector_search(limit=50)` first in a fresh transaction | **40** (index scan) |
| `vector_search(limit=50, ef_search=200)` | 50 |
| `vector_search(limit=50)` again, same transaction | **50** |
| `vector_search(limit=200)` again, same transaction | **200** |
| `SHOW hnsw.ef_search` at that point | **200** |
| `vector_search(limit=50)` in a new transaction | **40** |

`SET LOCAL` lasts until the end of the transaction. `fusion_heldout.py` runs
the chosen configuration (`ef_search=200`) and then the baseline (`ef_search`
unset) inside **one** `session_scope`, so the baseline inherited
`hnsw.ef_search = 200`. At ef=200 the planner does choose a sequential scan
(Phase 1 already recorded that), which is where the "Seq Scan at LIMIT 50 and
200, vector_candidates = 50" observation came from. In a fresh transaction the
production query uses the HNSW index and returns 40 rows at every limit from
50 to 200. Confirmed with EXPLAIN ANALYZE three times on the production query
form.

Consequences:

1. The G4 "baseline" is not the production configuration. It had 50 vector
   candidates; production has 40. The v3 frozen baseline's held-out `hybrid`
   nDCG is 0.913, the G4 baseline reads 0.916, and that difference is the leak.
2. Commit cfdd6b7 ("Close the Phase 1 ef_search question") is wrong and must
   be reverted in prose. The false claim lives in: `docs/results.md` lines
   ~101 and ~425, the `fusion_heldout.py` comment and protocol note, the
   Phase 4 lab-notes paragraph, and the committed
   `phase-8-fusion-heldout.json` `protocol.baseline_note`. Per A3 the evidence
   file is not edited; it is superseded (see F3) and its note gets a
   one-line correction in results.md.
3. `vector_search` has a design bug: any `SET LOCAL` it issues (ef_search,
   and also `enable_indexscan`/`enable_bitmapscan` under `exact=True`)
   persists for every later call in the same transaction. `hnsw_study.py`
   was safe only because it opens a session per call.

## Required fixes, in order

**F2. Make `vector_search` leak-proof** (own commit, no default change):
- When `ef_search is None`, issue `SET LOCAL hnsw.ef_search TO DEFAULT`
  explicitly, so a call never inherits a previous call's value.
- When `exact` is False, issue `SET LOCAL enable_indexscan TO DEFAULT` and
  `SET LOCAL enable_bitmapscan TO DEFAULT` for the same reason.
- Add an integration test (skipped without a DB) that calls
  `vector_search(limit=50, ef_search=200)` then `vector_search(limit=50)` in
  one session and asserts the second returns 40. Add the mirror test for
  `exact=True` followed by `exact=False` (plans differ; assert via row order
  equality is not enough, assert on `EXPLAIN` containing `Index Scan`).
- Add the rule to the brief, Part A, as A12b: "a `SET LOCAL` issued by a
  helper persists for the rest of the transaction; every helper must set
  every GUC it touches on every call, to a value or to DEFAULT."

**F3. Re-run G4 with three rows in one new evidence file**
(`phase-8-fusion-heldout-v2.json`; keep the old file, mark it superseded):
- `production-as-is`: k=60, pool 50, `ef_search` unset in a **fresh
  transaction per configuration** (or run it first). Expect 40 candidates
  per row; the verifier must assert `vector_candidates == 40`.
- `production-ef-fixed`: k=60, pool 50, `ef_search=50`. This isolates the
  effect of the cap alone.
- `chosen`: k=10, pool 200, `ef_search=200`, unchanged.
Run each configuration in its own `session_scope`. Report the same per-type
table with counts.

**F4. Revert the prose in cfdd6b7** and rewrite the Phase 1 closing sentence
to: "Confirmed in Phase 4: with `ef_search` unset the production query uses
the index and returns 40 candidates regardless of `candidate_limit`; the cap
is real. Whether to set `ef_search` explicitly in `retrieve()` is a G6
decision."

**F5. G6 decision items for Mayank** (after F3 numbers exist):
- (a) Set `ef_search = max(candidate_limit, 40)` inside `retrieve()`. This is
  a correctness fix that makes `candidate_limit` mean what it says; it does
  change production results. Reviewer recommends **yes**, as its own commit,
  citing the F3 file.
- (b) RRF `k`. Reviewer recommends **keep k=60** unless the F3 re-run shows
  the chosen configuration beating `production-ef-fixed` on held-out `all`
  by more than the topical-only movement seen so far. +0.005 on 26 questions
  with all movement in 6 topical questions is not evidence (A11).

## Small

- The sweep's cl50 cells used `ef_search=50`, not production's 40. Say so in
  results.md next to the sweep table; it is already in the protocol.
- `fusion_sweep.py` and `fusion_heldout.py` both pass `ef_search` explicitly
  on every call, so the sweep itself was not contaminated. State that in the
  lab notes so the leak finding is not over-read.
- Phase 3 results.md: the BM25 topical row is a lower bound and is marked;
  also mark that BM25's development `all` (0.887) includes that lower-bound
  topical component, so the honest headline is the paraphrase gap, not `all`.

## Not required

- No human has reviewed any Phase 3-4 ordering; the docs say so.
- The `ce_margin`/AUROC note from the last review is present.
