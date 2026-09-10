# PaperTrail retrieval upgrade — execution brief for Phases 1 through 5

Audience: the model executing this work (currently Claude Opus 4.8 in Claude
Code), starting from commit `8f18e1e` (Checkpoint 0 closed). Written by a
reviewing session on 2026-09-05. Read the whole brief before running anything.
This brief supersedes the phase descriptions in `retrieval-upgrade-workbook.md`
where they differ.

The owner is Mayank Ghadia. He reviews at each checkpoint. He is not watching
in real time. When this brief says STOP, stop and write a short status in
`docs/lab-notes.md` instead of guessing.

---

## Part A — Operating rules (read twice)

These are the failure patterns that a capable but hurried model produces on
this repo. Each rule exists because of a concrete incident or near miss.

**A1. Never run retrieval against the held-out split before a freeze, and never
tune anything on held-out.** On 2026-09-05 a dry-run touched the draft held-out
split and the entire split had to be replaced. Held-out is for one final
report per study. If you need to look at behaviour, use development questions.

**A2. Never edit a question file after its `frozen_at_utc` is set.** If labels
must change, create `questions-v3.1.json` with a new freeze, re-run, and keep
the old evidence file. The v1 keyword failure is still in `docs/evidence/`
for this reason.

**A3. Never edit an evidence JSON by hand, and never overwrite one.** Every run
writes a new file. If a run was wrong, keep it and write a second one with a
different name and a one-line note in the lab notes saying why. The verifier
exists to catch hand edits; do not try to satisfy it by editing summaries.

**A4. Report numbers by reading them from the evidence file, not from memory
of the terminal.** Before writing a table into the notes or README, run a
Python one-liner that prints the numbers from the JSON, and paste those. A
wrong digit in a README is an honesty failure even when accidental.

**A5. "Measured" and "expected" are different words.** If you did not run it,
write "not measured". Never write a number you predicted. If a run failed
half way, say which rows exist.

**A6. No silent fallbacks.** Any `try/except` that substitutes a different
model, a different index path, or a default value must either re-raise or
record the substitution in the output row. The `CrossEncoderReranker` once
silently fell back to a lexical scorer while still emitting `rerank_score`;
that pattern is banned. `except Exception: pass` is banned.

**A7. Do not change a default in the same commit as the study that justifies
it.** Study commit first (evidence + notes). Default change second, with the
evidence path in the commit message. This keeps each commit reviewable.

**A8. The package is not installed in editable mode.** After editing anything
under `src/`, run `python -m pip install '.[dev,ml]'` before running the CLI,
or the CLI runs stale code and you will report stale results without knowing.
Add this to your pre-run checklist for every phase. (Alternatively run
`python -m pip install -e '.[dev,ml]'` once and note it in the lab notes.)

**A9. Verify the database before every study.** Docker Desktop has shut down
between sessions and dropped the connection mid-run. Start each session with:

```bash
docker compose up -d db
papertrail verify-manifest docs/evidence/corpus-manifest-1000.json
```

and confirm 1,000 IDs match and 2,039 chunks are embedded. If a study's run
was interrupted, discard the partial output and rerun from scratch; do not
stitch partial runs.

**A10. Latency numbers need care.** The harness discards one warm-up per mode.
The first query after process start also pays model load. Never compare
latencies across runs on different days; only within one evidence file. Report
p50 and p95 with the row count.

**A11. Small n means small claims.** Held-out topical has 6 questions. Held-out
lexical has 8. Write "on the 6 held-out topical questions" and never "hybrid
hurts topical retrieval". The development split (12 topical, 24 paraphrase,
16 lexical) is where patterns can be described; held-out is where a chosen
configuration is confirmed once.

**A12. Postgres session settings.** `SET LOCAL` only applies inside the
current transaction. With SQLAlchemy `Session`, issue it via
`session.execute(text("SET LOCAL ..."))` in the same transaction as the query,
and do not `commit()` in between. Verify it took effect by reading it back with
`SHOW hnsw.ef_search` in the same transaction the first time you implement it.

**A12b. A `SET LOCAL` issued by a helper persists for the rest of the
transaction.** It is not scoped to the next statement. So every helper that
touches a GUC must set **every** GUC it touches on **every** call — either to a
value or explicitly back to `DEFAULT`. Otherwise a later call in the same session
silently inherits an earlier call's setting.

This is not hypothetical: on 2026-09-09 `fusion_heldout.py` ran the chosen
configuration (`ef_search=200`) and then the baseline (`ef_search` unset) inside
one `session_scope`. The baseline inherited `hnsw.ef_search = 200`, returned 50
vector candidates instead of the production 40, and produced a wrong conclusion
that the Phase 1 truncation "does not bind in production". A study that opens one
session per call (like `hnsw_study.py`) is accidentally safe; do not rely on that.

**A13. pgvector HNSW facts you will otherwise get wrong.**
- HNSW returns at most `hnsw.ef_search` rows. Default is 40. If you ask
  `LIMIT 50` with `ef_search = 40` you silently get 40 rows. Always set
  `ef_search >= limit`.
- `ef_search` is capped at 1000.
- Index build parameters `m` and `ef_construction` are baked at index creation
  (`models.py` sets m=16, ef_construction=64). Changing them means dropping
  and recreating the index, and afterwards you must recreate the original so
  the frozen evidence stays reproducible.
- Exact search is obtained by `SET LOCAL enable_indexscan = off` (and, to be
  safe, `enable_bitmapscan = off`). Confirm with `EXPLAIN` the first time that
  the plan shows a Seq Scan and not an Index Scan.

**A14. Paper-level vs chunk-level.** `vector_search` returns chunks;
`distinct_papers` collapses to one paper each. Recall studies for the index
must be done at the chunk level (that is what the index returns). Retrieval
metrics are at the paper level. Do not mix them in one table.

**A15. Tests must test the real code path.** A test that mocks the function
under test and asserts the mock was called proves nothing. For SQL paths,
unit-test the pure functions (metrics, fusion, thresholding, AUROC) with
hand-computed answers, and integration-test the SQL path only when the DB is
available (skip with a clear reason otherwise, as `test_embedding_search.py`
already does).

**A16. Authorship honesty.** Anything Claude authored (queries, grades,
adjudication, code) is labeled as such in the evidence protocol and in the
README. Never write "hand-curated" or "manually annotated" for Claude work.
Mayank's review of an artifact is "reviewed by Mayank", not "authored by".

**A17. Commit discipline.** One study per commit. Commit message names the
evidence file. Do not push. Do not squash earlier commits. Do not amend.

**A18. When to STOP and write a status instead of continuing.**
- A regression: the frozen v2 or v3 evidence no longer verifies.
- A study result that would change a default (gate signal, fusion weights,
  embedding model). Present the table; Mayank decides.
- Anything that needs a corpus change, a new question set, or a schema change
  beyond what a phase below explicitly specifies.
- Any test you cannot make pass without weakening it.

**A19. The pre-run checklist, every phase:**
1. `git status` clean, on a fresh commit.
2. DB up, manifest verified (A9).
3. Package reinstalled into the canonical `.venv-ml` (A8; it carries the `ml`
   extra). Print `pip show papertrail-rag` (name / version / location) into the
   lab-notes so a stale non-editable install cannot silently run old code (review
   F5, 2026-09-07).
4. `python -m pytest -q` green. Record the count.
5. Frozen v2 and v3 evidence verify:
```bash
papertrail verify-evidence --kind retrieval --report docs/evidence/phase-6-retrieval-evaluation-v2.json --manifest docs/evidence/corpus-manifest-1000.json
papertrail verify-evidence --kind retrieval --report docs/evidence/phase-8-retrieval-v3-baseline.json --manifest docs/evidence/corpus-manifest-1000.json --questions eval/questions-v3.json
```
6. Lab-notes entry opened with the date and phase.

**A20. The post-run checklist, every phase:**
1. New evidence file written under `docs/evidence/phase-8-<topic>.json`.
2. Verifier extended if the file has a new shape; verifier passes on it.
3. Step 5 of A19 re-run: old evidence still verifies.
4. Tests green, count recorded.
5. Lab notes: what ran, the table printed from the JSON, one design choice,
   one failure mode, anything surprising.
6. `docs/results.md` gets the table. README gets at most three sentences.
7. Commit.

---

## Part B — Two fixes before Phase 1 (30 minutes)

**B1. README adjudication wording.** `README.md` line ~71 says topical grades
had "every disagreement hand-adjudicated" without saying by whom. The
adjudicator was Claude Opus 4.8 (recorded correctly in the evidence
`protocol.topical_grade_provenance.adjudicator`). Change the README sentence to:

> Topical grades came from two independent model gradings (Cohen's kappa
> 0.719); a third model pass adjudicated every disagreement, and the record is
> in `eval/tools/adjudication.md`.

If Mayank has read and agreed with the adjudication record, append "reviewed by
Mayank Ghadia on <date>". Do not write that unless he has told you so.

**B2. Soften the held-out topical claim.** In README and `docs/results.md`,
wherever fusion is said to score below vector on held-out topical, add the
count: "on the 6 held-out topical questions". Keep the numbers.

Commit both as "Attribute adjudication and qualify the held-out topical
finding". No evidence changes.

---

## Part C — Phase 1: HNSW recall study

**Purpose.** `docs/how-it-works.md` states that no index-recall claim is made.
Make one, with numbers. Secondary: find out whether an approximate index is
even worth it at 2,039 vectors.

**Definition of recall here.** For one query embedding, `exact_k` = the set of
chunk IDs returned by an exact (sequential-scan) cosine search with `LIMIT k`,
and `approx_k` = the same from the HNSW index. Chunk-level recall@k =
`|approx_k ∩ exact_k| / k`. Use k = 10 and k = 50. Ties at the boundary can
cause recall slightly below 1.0 even for a perfect index; note that in the
write-up rather than "fixing" it.

**C1. Code changes** (`src/papertrail/repository.py`):

```python
def vector_search(
    session, query_embedding, *, limit, exact: bool = False, ef_search: int | None = None
):
    if exact:
        session.execute(text("SET LOCAL enable_indexscan = off"))
        session.execute(text("SET LOCAL enable_bitmapscan = off"))
    elif ef_search is not None:
        if not 1 <= ef_search <= 1000:
            raise ValueError("ef_search must be in [1, 1000]")
        if ef_search < limit:
            raise ValueError("ef_search must be >= limit or results are truncated")
        session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    ...existing query...
```

Do not use string formatting for anything but the validated integer. Add a
docstring stating A12/A13.

Verify once by hand in a Python shell that `EXPLAIN` shows `Seq Scan` under
`exact=True` and `Index Scan using ix_chunks_embedding_hnsw_cosine` otherwise.
Paste both plans into the lab notes.

**C2. The study script** `eval/tools/hnsw_study.py`:

- Load `eval/questions-v3.json`. Use ALL retrieval questions, both splits.
  This is allowed because nothing is being selected; the index parameters
  stay at their defaults after the study. State this reasoning in the file
  docstring.
- Encode each query once with the production encoder.
- For each query: `exact_50 = vector_search(..., limit=50, exact=True)`.
  Then for `ef in (10, 40, 100, 200, 400, 1000)`:
  `approx = vector_search(..., limit=50, ef_search=ef)` and compute
  recall@10 and recall@50 against `exact_50` (use the first 10 of each for
  recall@10). Record wall time of each call with `time.perf_counter()`.
- Discard the first query's timings as warm-up (run it twice, keep the second).
- Write `docs/evidence/phase-8-hnsw-recall.json`:

```json
{
  "schema_version": 1,
  "created_at_utc": "...",
  "evaluation_set_frozen_at_utc": "<copy from questions-v3.json>",
  "corpus_arxiv_ids_sha256": "...",
  "embedding_model": "...",
  "index": {"type": "hnsw", "m": 16, "ef_construction": 64, "ops": "vector_cosine_ops"},
  "protocol": {"k_values": [10, 50], "ef_search_values": [10, 40, 100, 200, 400, 1000],
               "queries": 78, "level": "chunk", "warmup_discarded": true},
  "rows": [ {"question_id": "...", "ef_search": 40, "recall_at_10": 1.0, "recall_at_50": 0.98,
             "approx_latency_ms": 3.1, "exact_latency_ms": 4.7} ],
  "summary": { "ef_search": { "40": {"mean_recall_at_10": ..., "mean_recall_at_50": ...,
               "p50_latency_ms": ..., "p95_latency_ms": ...} },
               "exact": {"p50_latency_ms": ..., "p95_latency_ms": ...} },
  "claim_boundary": "Chunk-level index recall on 2,039 vectors with one query set; not a general HNSW benchmark."
}
```

**C3. Verifier.** Add `verify_hnsw_evidence(path, manifest)` in `evidence.py`
and a `--kind hnsw` choice in the CLI. It recomputes every `summary` value
from `rows`, checks the corpus hash, checks freeze-before-created, and checks
that every `ef_search` in rows is `>= 50` OR that the row's recall@50 was
computed against a truncated list and says so. Simplest: require rows with
`ef_search < 50` to carry `"truncated": true` and compute recall@50 with
denominator 50 anyway (so ef=10 shows recall@50 ≤ 0.2 honestly).

**C4. Tests** (`tests/test_evaluation.py` or a new `tests/test_hnsw_study.py`):
- Pure function `chunk_recall(approx_ids, exact_ids, k)` with a hand-computed
  answer, including the truncated case.
- `vector_search` raises when `ef_search < limit`.
- Verifier rejects an edited summary (copy the pattern of
  `test_retrieval_verifier_rejects_edited_summary`).

**C5. Optional index-parameter sweep.** Only if C2 shows recall@50 below 0.99
at ef=100. Otherwise skip and say so. If done: for `(m, ef_construction)` in
`[(16,64), (16,200), (32,200)]`, drop and recreate the index, rerun C2 into a
separate file `phase-8-hnsw-recall-m<M>-efc<EFC>.json`, and finally recreate
the original `(16,64)` index and re-verify the v3 baseline still verifies
(retrieval results must be identical; if they are not, HNSW build
nondeterminism is the finding, record it and STOP).

**C6. Definition of done.** Evidence file verified; table in notes printed from
JSON; a sentence in `how-it-works.md` replacing "no index-recall claim is
made" with the measured recall at the default ef_search=40 and the exact-scan
latency comparison. If exact scan is faster or equal at this corpus size, say
so; that is the honest and interesting result.

**Traps specific to this phase.**
- Forgetting that the default `ef_search` is 40 while `candidate_limit` in
  `retrieve()` is 50 to 200. This means production hybrid retrieval today may
  be silently truncated to 40 vector candidates. Check this. If true, it is a
  real finding: record it, do not fix it in this phase (A7), and flag it for
  Phase 4 where candidate pools are studied.
- Comparing chunk recall with paper metrics (A14).
- Running the study with the package not reinstalled (A8).

---

## Part D — Phase 1b: Abstention gate redesign

**Purpose.** The gate refused 2 of 10 answerable v2 held-out questions because
its signal (top RRF score) is quantized by rank position, not by match
quality. Replace it with a measured better signal, and make refusals useful.

**Background numbers to keep in mind** (from the v2 evidence): the RRF top
score takes values like 0.03279 (rank 1 in both lists), 0.03252 (ranks 1 and
2), 0.03178 (ranks 1 and 5). The frozen threshold 0.03239 sits between the
second and third. A top-1 vector hit was refused because the keyword list
ranked it fifth.

**D1. Signal collection script** `eval/tools/gate_signals.py`.

For every v3 retrieval question AND every v3 abstention question, both splits
(collection is not selection; selection happens in D2 on development only):

| Signal | Computation |
|---|---|
| `rrf_top` | `retrieve(mode="hybrid", limit=5)[0]["score"]` (current gate) |
| `cos_top` | `vector_search(limit=50)[0]["score"]` (already `1 - cosine distance`) |
| `cos_margin` | `cos_top - vector_search(...)[1]["score"]` |
| `cos_mean_top3` | mean of the top-3 vector scores |
| `kw_top` | `keyword_search(limit=50)[0]["score"]` or `0.0` if no rows |
| `ce_top` | `retrieve(mode="hybrid_rerank", limit=5)[0]["rerank_score"]` |
| `ce_margin` | top minus second `rerank_score` |
| `ce_sigmoid_top` | `1/(1+exp(-ce_top))` (the ms-marco model emits logits; record both) |

Each row: `question_id, split, type, is_answerable (bool), <signals>`.
Also record `retrieved_rank_of_relevant` for answerable questions (the rank of
the best-graded relevant paper in the hybrid list, or null), so the write-up
can distinguish "refused although retrieved at rank 1" from "refused because
not retrieved".

Write `docs/evidence/phase-8-gate-signals.json` with `rows`, protocol, hash,
freeze timestamp, and the reranker model name actually loaded (A6: assert it
is the cross-encoder, not a fallback; the reranker now raises without
`allow_fallback=True`).

**D2. Scoring signals** — add to `evaluation.py`:

```python
def auroc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Rank-sum AUROC with tie handling (ties count 0.5)."""
```

Implement by counting pairs (n ≤ 100 × 30, so O(n·m) is fine). Test with
hand-computed cases: perfect separation → 1.0, reversed → 0.0, identical
lists → 0.5, one tie → check by hand.

Then, on **development rows only**, for each signal: AUROC, and the
balanced-accuracy threshold via the existing `_choose_threshold`. Write
`docs/evidence/phase-8-gate-selection.json` with per-signal dev AUROC, dev
threshold, dev positive-accept and negative-abstain rates, and the chosen
signal (highest dev AUROC; tie-break by dev balanced accuracy; then prefer
the cheaper signal: cosine over cross-encoder).

**D3. Held-out report.** Apply each signal's frozen dev threshold to held-out
rows once. Report per signal: false-refusal rate on answerable, false-answer
rate on negatives (split into `negative_ood` and `negative_near`), balanced
accuracy. Put this in the same file under `heldout`. Also report, for the
current `rrf_top` gate, how many answerable held-out questions are refused
although the relevant paper is at hybrid rank 1 or 2. That number is the
headline.

**D4. Verifier** `--kind gate`: recompute AUROCs and thresholds from rows,
check chosen-signal rule, check freeze chronology, check the reranker model
name is present and not a fallback name.

**D5. Code change to the gate** (separate commit, only after Mayank has seen
the D3 table — A18):

- `config.py`: `abstain_signal: str = "rrf_top"` and keep `abstain_threshold`.
- `generation.py`: compute the configured signal instead of reading
  `hits[0]["score"]`. Factor the signal computation into a small function
  `gate_signal(session, question, encoder, signal_name) -> float` that
  `gate_signals.py` also uses, so the study and production compute the same
  number. `AnswerResult` gains `gate_signal_name`.
- Default stays `rrf_top` in this commit.

**D6. Soft abstention** (can be in the same commit as D5):

- On abstention, `AnswerResult.nearest_papers` holds the top-3 retrieved
  `(arxiv_id, title, source_url)`. No generation call is made.
- API and CLI show them under "Low confidence. Closest evidence:".
- Test: abstention result has 3 nearest papers and `answer is None`; a
  non-abstention result has an empty `nearest_papers`.

**D7. Two-gate RAG evaluation** (needs `GROQ_API_KEY`; if absent, STOP after
D6 and say so):

Extend `evaluate-rag` with `--gate-signal <name>` and `--verify-entailment`.
Run held-out three times, three separate evidence files:
1. current gate (`rrf_top`, frozen threshold) — the baseline;
2. chosen signal from D2 with its frozen dev threshold;
3. chosen signal + entailment check (`min_faithfulness=0.80`).

Report for each: answerable answered / refused (and refused-at-rank-1 count),
negatives refused / answered, entailment refusals, provider errors, citation
grounding rate. Same verifier rules as the existing RAG verifier.

Groq rate limits: keep the bounded retry; if a run records provider errors,
keep the file and rerun into a new file, as was done for v2.

**D8. Default change** (separate commit): only if the chosen signal improves
held-out balanced accuracy AND does not increase the false-answer rate on
`negative_near`. If it trades one for the other, STOP and present both.

**Definition of done.** Signals file, selection file, RAG comparison files,
all verified; notes explain the rank-quantization bug in plain words with the
actual numbers; README paragraph on abstention rewritten to describe the new
gate only if the default changed.

**Traps specific to this phase.**
- Picking the signal on held-out numbers. Selection is D2 on development;
  D3 is a single report.
- Reranker fallback silently producing `ce_top` from lexical scoring (A6).
- The cross-encoder outputs logits that can be negative; do not clamp them
  to [0,1] before thresholding, and do not compare `ce_top` thresholds with
  `cos_top` thresholds; only AUROC and rates are comparable across signals.
- `threshold` in `answer_question` is validated to `[0,1]`; a logit signal
  breaks that check. Move the range check into the signal definition (each
  signal declares its valid range) rather than deleting the check.
- Balanced accuracy hides the asymmetry; always print both error rates.
- Changing the default gate in the study commit (A7).

---

## Part E — Phase 2: Reranker evaluation

**Purpose.** `hybrid_rerank` exists and led every v3 cell, but only at one
pool size with one model. Measure the pool-size and model trade-off, its
latency cost, and whether the reranker or the fusion did the work.

**E1. Parameterize** `retrieve()` with `rerank_pool: int | None = None`
(default keeps the current `max(limit * 2, 20)`), and add mode
`"vector_rerank"` that reranks the vector list alone (same pool size). Add
`SearchMode` literal, CLI choice, and a unit test that `vector_rerank` never
calls `keyword_search` (monkeypatch it to raise).

**E2. Models.** `CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")`
(current) and `CrossEncoderReranker("BAAI/bge-reranker-base")`. Confirm both
load with a printed score in the lab notes before the study (A6). If
`bge-reranker-base` needs `sentence_transformers.CrossEncoder` kwargs (it may
need `max_length=512`), set them explicitly and record them.

**E3. Study.** For each config in
`{ms-marco, bge} × {pool 20, pool 50} × {hybrid_rerank, vector_rerank}`,
run the v3 evaluation. The existing `evaluate()` runs all `MODES`; add an
`--modes` CLI option and a `--reranker`/`--rerank-pool` pair, and store them
in `protocol`. One evidence file per config:
`phase-8-rerank-<model>-pool<N>.json`. Do not touch the frozen baseline file.

**E4. Report** (development split for the pattern, held-out once for the
confirmation): per-type nDCG@10, Recall@10, p50/p95 latency, versus the v3
baseline `hybrid` and `hybrid_rerank` rows. The `vector_rerank` vs
`hybrid_rerank` comparison answers "did fusion or the reranker do the work".

**E5. Verifier.** The v3 retrieval verifier must accept a `protocol.modes`
subset and require `protocol.reranker_model` and `protocol.rerank_pool` when
any rerank mode is present.

**Definition of done.** Eight verified files, one comparison table in notes
and `results.md`, and a stated recommendation with the latency cost attached.
No default change in this phase.

**Traps.**
- Model download during a latency-measured run. Load both models before the
  timed loop.
- `hybrid_rerank` pool defaults to 20 for `limit=10`; if pool 50 hits the
  `distinct_papers` cap because `candidate_limit` is only 50 chunks per
  side, the pool is smaller than requested. Record the actual pool size per
  row (`len(fused_hits)` before rerank).
- Treating a +0.02 nDCG on 6 held-out topical questions as a result (A11).

---

## Part F — Phase 3: Sparse retrieval

**Purpose.** Keyword mode is the weakest component (dev nDCG 0.759; 0.695 on
paraphrase; 0.497 on held-out topical). It is OR-of-all-terms ranked by
`ts_rank_cd` with no length normalization and no field weighting.

**F1. Offline BM25 ablation** (`eval/tools/bm25_ablation.py`, no DB writes):

```bash
python -m pip install rank_bm25
```

- Read all chunks (`SELECT id, arxiv_id, text FROM chunks`).
- Tokenize with the same regex as `keyword_search` plus lower-casing. Do not
  stem (Postgres `english` config stems; note the difference in the write-up
  rather than trying to match it exactly).
- `BM25Okapi(tokens)`; for each v3 retrieval query, take top-200 chunks,
  collapse to distinct papers with the existing `distinct_papers`, keep 10,
  compute the same per-row metrics (`recall_at`, `ndcg_at` graded).
- Write `docs/evidence/phase-8-bm25-offline.json` with rows and per-type
  aggregates, `library: rank_bm25 <version>`, `k1`, `b` defaults recorded.
- Compare against the `keyword` rows in the v3 baseline file, development
  split, per type.

Decision rule, written into the notes before looking at results: if BM25
beats Postgres FTS by ≥ 0.05 nDCG@10 on development `lexical` or `paraphrase`,
the ranking function matters (do F2-iii). If the gap is < 0.05, the query
construction matters more (do F2-i first).

**F2. Postgres-side changes**, each its own commit and evidence file
(`phase-8-keyword-<variant>.json`, run with `--modes keyword`):

- **i. AND-then-OR cascade.** In `keyword_search`, build
  `to_tsquery('english', 'a & b & c')`. If it returns fewer than `limit` rows,
  append rows from the existing OR query that are not already present,
  keeping AND rows first. Expose the old behaviour behind
  `keyword_strategy="or"` for A/B; default remains `"or"` until Phase 4
  chooses (A7).
- **ii. Phrase boost.** If the query contains a quoted span, or two adjacent
  tokens that both start with a capital letter in the original query, add a
  `phraseto_tsquery` match set ranked first. Small, may do nothing on this
  question set; measure and say so.
- **iii. Field weights and length normalization.** New Alembic migration
  `20260906_0003_weighted_search_vector.py` replacing the generated column
  with

  ```sql
  setweight(to_tsvector('english', split_part(text, E'\n\n', 1)), 'A') ||
  setweight(to_tsvector('english', substr(text, length(split_part(text, E'\n\n', 1)) + 3)), 'B')
  ```

  (the chunk text is `title + "\n\n" + abstract_chunk`; confirm with a
  `SELECT` on three rows before writing the migration). Rank with
  `ts_rank_cd('{0.1, 0.2, 0.4, 1.0}', search_vector, query, 32)`. The `32`
  flag divides by document length. Downgrade path must restore the original
  expression exactly. Run `alembic downgrade -1` then `upgrade head` once to
  prove both directions work, and re-verify the v3 baseline afterwards (it
  must still verify; if keyword results changed under the *old* strategy,
  something else changed; STOP).
- **iv. ParadeDB `pg_search`.** Only if F1 says the ranking function matters
  by a wide margin AND iii did not close the gap. It is a second Compose
  service and a different image. Do not add it to the default stack.

**F3. Report.** Per-type development table: FTS-OR (baseline), BM25 offline,
cascade, cascade+phrase, cascade+weights. Held-out once for the best
development configuration. Recommendation to Phase 4 only.

**Traps.**
- `to_tsquery` raises on stop-word-only input ("the of and"). Guard it: if
  the AND query text is empty after stop-word removal, skip to OR. Test it.
- Hyphenated tokens: the regex keeps `state-of-the-art` as one token, and
  `to_tsquery` will parse `-` as NOT unless quoted. Quote each term
  (`'''term'''`) or strip hyphens consistently; test with a hyphenated query.
- The GIN index must be rebuilt by the migration; check `\d chunks` shows it.
- Stemming: FTS stems, BM25 offline does not. State it; do not "fix" BM25 to
  match, that is a different study.

---

## Part G — Phase 4: Fusion ablation

**Purpose.** RRF `k=60`, equal weights, candidate pool 50 to 200 per side were
never ablated. Also the Phase 1 trap: `hnsw.ef_search` default 40 may be
truncating the vector candidate list below `candidate_limit`.

**G1. Parameterize.**
- `reciprocal_rank_fusion(rankings, k=60, weights=None)` multiplies each
  source's `1/(k+rank)` by `weights[source]` (default 1.0). Existing tests
  must pass unchanged with `weights=None`.
- `convex_fusion(rankings, alpha)`: min-max normalize each source's `score`
  to [0,1] within its own list, then `alpha * vec + (1 - alpha) * kw`, missing
  source contributes 0. Deterministic tie-break identical to RRF. Unit test
  with a hand-computed three-document example.
- `retrieve(..., rrf_k, weights, fusion="rrf"|"convex", alpha, candidate_limit, ef_search)`.

**G2. Sweep script** `eval/tools/fusion_sweep.py`, **development split only**:

- `k ∈ {10, 30, 60, 100}`, `w_vec ∈ {1, 2, 3}` with `w_kw = 1`;
- `alpha ∈ {0.5, 0.7, 0.9}` for convex;
- `candidate_limit ∈ {50, 200}` with `ef_search` set to
  `max(candidate_limit, 40)` so vector candidates are not truncated (record
  the value);
- keyword strategy: whichever Phase 3 recommended, plus the baseline OR.
- One row per (config, question, mode="hybrid"). Store everything in
  `docs/evidence/phase-8-fusion-sweep-dev.json`.

**G3. Selection rule**, written into the notes before running: the
configuration with the highest development nDCG@10 on `all`; ties broken by
`paraphrase` then `lexical`; among remaining ties, the simplest (unweighted
RRF over weighted, smaller pool over larger).

**G4. Held-out confirmation**: run the chosen configuration and the v2-style
baseline (`k=60`, equal, pool 50, ef 40, OR keyword) on held-out once each,
`phase-8-fusion-heldout.json`, per type, with the counts.

**G5. Verifier.** `protocol.rrf_k` is compared to the report's own declared
protocol, not to the constant 60, for schema-3 fusion files; the verifier
must also check that the held-out file's configuration equals the dev-best in
the sweep file (read both).

**G6. Default change**, separate commit, only after Mayank sees G4 (A18).

**Traps.**
- Min-max normalizing over the truncated list makes the last candidate
  exactly 0; that is expected, document it.
- Changing `candidate_limit` without `ef_search` re-introduces truncation.
- Reporting the best sweep cell as "the result": the sweep is development;
  only G4 is reportable as a result.

---

## Part H — Phase 5: Embedding model ablation

**Purpose.** MiniLM-L6 (384-d) is the only encoder ever tried.

**H1. Constraints in the code you must respect.**
- `EMBEDDING_DIMENSIONS` is a module constant and
  `SentenceTransformerEncoder` raises if the model's dimension differs. Do not
  delete this check; make the expected dimension a constructor argument that
  defaults to the constant, and have `require_vector_search_ready` compare
  against the `embedding_runs` row for the chosen model.
- `chunks.embedding` is `VECTOR(384)`. A 768-d model needs a new column
  `embedding_768 VECTOR(768)` with its own HNSW index (migration
  `20260907_0004_add_embedding_768.py`). Keep the 384 column untouched.
- `embedding_runs` needs a `column_name TEXT NOT NULL DEFAULT 'embedding'`
  field (same migration) so a run says where its vectors live.
- `encode()` sets `normalize_embeddings=True`; keep it, and record
  `normalized=True` in the run row (already done).

**H2. Prefixes.** `SentenceTransformerEncoder(model_name, query_prefix="",
passage_prefix="")`. `embed` uses `passage_prefix`; `retrieve` and every study
script use `query_prefix`. Store both prefixes in the `embedding_runs` row
(add two TEXT columns in the same migration). Models and their prefixes:

| Model | dims | query prefix | passage prefix |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | none | none |
| `BAAI/bge-small-en-v1.5` | 384 | `Represent this sentence for searching relevant passages: ` | none |
| `intfloat/e5-small-v2` | 384 | `query: ` | `passage: ` |
| `thenlper/gte-small` | 384 | none | none |
| `BAAI/bge-base-en-v1.5` | 768 | same as bge-small | none |

Check each model card once and paste the relevant line into the lab notes; do
not rely on this table alone.

**H3. Procedure per model.**
1. `papertrail embed docs/evidence/corpus-manifest-1000.json --model <m> --column <col> --passage-prefix "<p>"`
   (extend the CLI; the 384-d models overwrite `embedding`, so run them one
   at a time and re-embed MiniLM last to restore the frozen state — OR, better,
   add per-model columns `embedding_bge_small`, etc. Choose the second option
   if disk allows; 2,039 × 384 floats is 3 MB per column, so it does).
2. Record wall time and `pg_relation_size` of the index.
3. `papertrail evaluate --questions eval/questions-v3.json --manifest ... --model <m> --column <col> --query-prefix "<q>" --modes vector,hybrid,hybrid_rerank --output docs/evidence/phase-8-embed-<slug>.json`.
4. Negative control: e5-small with NO prefixes, its own file
   `phase-8-embed-e5-small-noprefix.json`. Expect it to be worse; if it is
   not, say so.

**H4. After all models**: re-verify the v3 baseline file. It reads the
`embedding` column with MiniLM; results must be byte-identical. If you used
overwrite-and-restore instead of separate columns and the baseline no longer
verifies, the re-embedding was not deterministic; record it and STOP.

**H5. Report.** Model × per-type development nDCG@10 × embed wall time ×
index size × vector p95 latency. Held-out once for the best development
model. Default change only by Mayank's decision.

**Traps.**
- Forgetting the query prefix at query time while using it at index time
  (or vice versa). The negative control exists to show the size of this
  mistake; make sure the *main* runs do not make it.
- Mixing columns: `vector_search` must take the column as a parameter and the
  study must pass it explicitly. Add an assertion that the `embedding_runs`
  row for `(model, column)` is `complete` before evaluating.
- Running the reranker mode with a different encoder changes only the
  candidate list; say so when reporting `hybrid_rerank` per model.

---

## Part I — Reporting format for every phase

In `docs/lab-notes.md`, one entry per phase:

```
## 2026-09-0X — Phase N: <title>
Pre-run: tests <n> passed; v2 + v3 evidence verified; DB manifest verified; package reinstalled.
Ran: <commands>
Evidence: docs/evidence/phase-8-<...>.json (verified)
Table (printed from the JSON):
| ... |
Design choice: <one sentence, why this and not the alternative>
Failure mode found: <one sentence, with the number>
Surprises / not measured: <...>
Post-run: tests <n> passed; old evidence still verifies.
```

In `docs/results.md`: the table only, with counts per split and type.
In `README.md`: at most three sentences per phase, numbers copied from the
evidence file, claim boundary preserved.

---

## Part J — Order and rough budget

| Order | Part | Days | Blocking input from Mayank |
|---|---|---|---|
| 1 | B (two fixes) | 0.1 | none |
| 2 | C (HNSW) | 0.5 | none |
| 3 | D1–D4 (gate study) | 0.5 | none |
| 4 | D5–D6 (gate code, soft abstention) | 0.3 | Mayank sees D3 table before D8 |
| 5 | D7 (RAG two-gate) | 0.3 | `GROQ_API_KEY` |
| 6 | E (reranker) | 0.5 | none |
| 7 | F (sparse) | 1.0 | none |
| 8 | G (fusion) | 0.5 | Mayank sees G4 before default change |
| 9 | H (embeddings) | 1.0 | Mayank decides default |

Stop after any row; each leaves a complete, verified, committed result.
