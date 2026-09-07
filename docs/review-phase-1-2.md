# Review of Phases 1, 1b, 2 (commits 2caa786..8dc73ff) — 2026-09-07

Reviewer: Claude Fable 5.1 session, on behalf of Mayank Ghadia. Everything
below was checked by reading the evidence files and code, not the lab notes.

## Verified OK

- All 15 new evidence files verify with the current package (hnsw 467 rows,
  gate 105 rows, 8 rerank files × 78 rows, 3 RAG files × 35 records). Frozen v2
  (retrieval + RAG) and v3 baseline still verify. 64 tests pass.
- No Groq key in git history or tracked files.
- Held-out was used only for single reports; selection was on development.
- HNSW study: sound, `exact` and `ef_search` implemented per brief, EXPLAIN
  plans recorded, truncation flagged honestly, C5 sweep correctly skipped.
- Gate study: sound. Stopping before D8 on the near-miss trade-off was the
  right call. Signal computation is shared between study and production.
- Reranker study: sound. Pool-50 "leaky knob" and unjudged-paper lower bounds
  are recorded per row and recomputed by the verifier. Keeping the default is
  the right reading.
- Groq model retirement handled correctly: preserved 404 runs, separate
  default-change commit, stale doc claims fixed.

## Must fix before Phase 3

**F1. "NLI entailment" is a token-overlap heuristic, and the docs call it NLI.**
`entailment.evaluate_entailment` defaults to `HeuristicNLIJudge`, which is a
word-overlap + negation heuristic with no model. No NLI model exists in the
package. The D7 result "the statement-level NLI entailment gate at 0.80
refuses every answer" (results.md, lab notes, README line 30 "statement-level
NLI check") is therefore a result about an overlap heuristic. The evidence
file does not record which judge ran.
- Rename in docs: "heuristic token-overlap faithfulness check" everywhere
  "NLI" or "entailment model" appears (README, status.md, results.md, lab
  notes Phase 1b, commit-message history stays as is).
- Add `entailment_judge: "HeuristicNLIJudge (token overlap, no model)"` to
  the RAG evidence protocol for future runs; for the existing file, add a
  line in results.md saying the judge is not recorded in the file and was the
  heuristic (the only judge that exists).
- Rename the class to `HeuristicOverlapJudge` in a follow-up; keep the
  Protocol name `NLIJudge` only if a real NLI judge is added later.
- The finding itself is still useful: reword as "an overlap heuristic at 0.80
  refuses every gpt-oss-120b answer; a real NLI judge was never built."

**F2. Citation-format failures are under-reported.** With `gpt-oss-120b`,
5 of 26 (rrf_top) and 8 of 26 (cos_mean_top3) answerable held-out questions
raised "no citation" and are counted as neither answered nor abstained. The
v2 Llama run had zero enforcement errors, so this is a regression from the
model migration, not "working as designed". Required:
- results.md and README gate paragraph: state the no-citation counts next to
  the answer rates ("answered 13, refused 8, uncited 5 of 26").
- `answerable_answer_rate` must not be presented alone; add
  `answerable_uncited_rate` to the RAG summary and verifier.
- Open a follow-up task: model-specific citation prompt for gpt-oss-120b,
  measured on development questions only, before any further RAG evidence.

**F3. `answerable_refused_at_rank_1or2` conflates gate and heuristic
refusals.** In the entail file it reads 15, of which 1 is the gate. Split into
`gate_refused_at_rank_1or2` and `entailment_refused_at_rank_1or2`, recompute in
the verifier, and correct the results.md table cell (it currently shows 15
under "refuse@rank1-2", which reads as a gate number).

**F4. Latent gate bug.** `answer_question` computes `rrf_top` from
`hits[0]["score"]` for whatever `retrieval_mode` was passed. If a caller passes
`hybrid_rerank`, that score is a cross-encoder logit compared against 0.0324
and the gate is meaningless. Not triggered today (default mode is hybrid).
Fix: when `gate_signal_name == "rrf_top"` and `retrieval_mode != "hybrid"`,
raise, or compute the signal via `gate_signal()` which retrieves hybrid
itself. Add a test.

**F5. The `.venv` interpreter was stale.** The executing session ran from
`.venv-ml`; `.venv` (the one CLAUDE.md names for tests) still had the
pre-Phase-1 package, so `verify-evidence --kind hnsw` failed there until the
reviewer reinstalled it. Add to the lab notes which venv is canonical, and
add the reinstall to the pre-run checklist output (A19 step 3 should print
`pip show papertrail` version + install time).

## Should fix (small)

- results.md Phase 1 says exact scan "~10 ms" from EXPLAIN while the table
  says p50 23.1 ms including connection setup. Keep both but put the EXPLAIN
  number in the same sentence as its caveat.
- README "HNSW index recall" paragraph: "recall@50 0.998 once ef_search≥100"
  is fine; add "recall@50 is undefined below ef=50" so a reader does not read
  0.800 at ef=40 as approximation error.
- The polluted v3-baseline `hybrid_rerank` latency (dev p50 3896 ms) is still
  in the frozen file. Do not edit it (A3); add a one-line footnote in
  results.md v3 table pointing to the clean Phase 2 number.
- Gate study: `ce_margin` had the best held-out balanced accuracy (0.923)
  and zero near-miss answers but lost on dev AUROC. Note in results.md that a
  selection rule using near-miss cost would have chosen differently, as a
  limitation of AUROC-only selection, not as a new choice.

## Not required, worth noting

- The entailment refusal rows carry `citations: []` after the harness fix;
  good.
- No human has reviewed any Phase 1-2 ordering or label; the docs say so.
