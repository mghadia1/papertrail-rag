# PaperTrail retrieval upgrade — self-execution workbook

Companion to `retrieval-upgrade-plan.md`. That file says *what* and *why*; this
one says *how*, step by step, for someone who has read the codebase once and
can write Python and SQL. Each phase ends with a checkpoint and an interview
note. Do the phases in order. Commit after each checkpoint.

Conventions used below:

- `$PT` = `projects/papertrail-rag`. All commands run from there with
  `.venv` active.
- "Evidence file" = a JSON under `docs/evidence/` that `verify-evidence` can
  recompute. Never edit one by hand after it is created; make a new one.
- "Notebook" = `docs/lab-notes.md`, a running log you create in Phase S. One
  dated entry per work session: what you ran, what you saw, what surprised
  you. This is where interview answers come from.

---

## Phase S — Setup (half a day)

**S1. Get the database up and verified.**

```bash
open -a Docker        # wait for the whale icon to settle
docker compose up -d db
alembic upgrade head
papertrail verify-manifest docs/evidence/corpus-manifest-1000.json
```

If `verify-manifest` reports a mismatch the volume is empty; run
`papertrail replay-manifest docs/evidence/corpus-manifest-1000.json` (about 30
minutes with the polite 3 s delay) and then `papertrail embed
docs/evidence/corpus-manifest-1000.json`.

**S2. Install the ML extra and prove the cross-encoder loads.**

```bash
python -m pip install '.[dev,ml]'
python -c "from sentence_transformers import CrossEncoder; m=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print(m.predict([['what is RRF','Reciprocal rank fusion combines rankings']]))"
```

You want a number printed, not an exception. Write the number in the notebook.

**S3. Reproduce the frozen baseline before touching anything.**

```bash
papertrail evaluate --questions eval/questions-v2.json \
  --manifest docs/evidence/corpus-manifest-1000.json \
  --output /tmp/baseline-rerun.json
papertrail verify-evidence --kind retrieval --report /tmp/baseline-rerun.json \
  --manifest docs/evidence/corpus-manifest-1000.json
```

Compare the aggregates against `docs/evidence/phase-6-retrieval-evaluation-v2.json`.
Metrics must match exactly; latency will differ. If metrics differ, stop and
find out why before doing anything else (usually: incomplete embeddings or a
different model cached).

**S4. Create `docs/lab-notes.md`** with today's date and the three outputs above.

**Checkpoint S:** database verified, cross-encoder loads, baseline reproduces.

---

## Phase H — Honesty fixes (2 hours)

**H1. Make the reranker fallback explicit.** In `src/papertrail/reranking.py`:

- Add `allow_fallback: bool = False` to `CrossEncoderReranker.__init__`.
- In `_load_model`, on exception: if `allow_fallback` is false, re-raise with
  a message naming the model; otherwise set `self._model = False` and set
  `self.model_name = self._fallback.model_name` so the output is labeled
  honestly.
- In `rerank`, add `item["reranker"] = self.model_name` to every result.

Update `tests/test_reranking.py::test_cross_encoder_fallback_works_when_model_absent`
to pass `allow_fallback=True`, and add a test that the default raises when
the model cannot load (monkeypatch the import to fail).

**H2. Expose `hybrid_rerank` on the CLI.** In `cli.py` the `search --mode`
choices are hard-coded to three values; add `hybrid_rerank`. Run
`papertrail search "diffusion planning" --mode hybrid_rerank` and confirm the
`reranker` field shows the real model name.

**H3. Fix the docs.** README: test count 48 → whatever `pytest` prints now;
add one sentence under *Verified system*: "A `hybrid_rerank` mode and a
statement-level NLI check exist in code; neither has a published metric."
Same sentence in `docs/status.md`.

**Checkpoint H:** tests green, README accurate, `search --mode hybrid_rerank`
shows a real model name. Commit: "Make reranker fallback explicit and document
unevaluated modes".

**Interview note:** "I found a silent fallback that could have mislabeled
evidence. I made it opt-in and stamped the real model name on every row."

---

## Phase 0 — Evaluation v3 (2 days, mostly labeling)

The v2 set is saturated. This phase is the foundation for everything after.

**0a. Understand the hard-coded assumptions you are about to break.**
`evidence.verify_retrieval_evidence` requires exactly 90 rows and 20/10
dev/held-out per mode; `evaluation.MODES` is a fixed 3-tuple; the question
file has a flat `relevant_arxiv_ids` list with no grades. Write these three
facts in the notebook. You will change all three.

**0b. Sample papers for labeling.** Write `eval/tools/sample_papers.py`:

```python
# select 90 papers uniformly at random with a fixed seed from the manifest,
# print arxiv_id, title, abstract to eval/tools/sample.md for reading
```

Use `random.Random(20260903)`. Record the seed in the question file later.

**0c. Write the queries.** Open `sample.md` and for each paper write one query
of a given type. Aim for this mix (dev / held-out):

| Type | Count | Rule |
|---|---|---|
| `paraphrase` | 24 / 12 | Describe the method or result in your own words. **No word from the title may appear in the query.** Enforce this with a script, not by eye. |
| `lexical` | 16 / 8 | Build the query around a proper noun, acronym, dataset, or number from the abstract. Include at least one term that a MiniLM paraphrase would miss. |
| `topical` | 12 / 6 | "Papers about X" where you expect 2–6 hits in the corpus. Relevance is decided by pooling in 0e. |
| `negative_ood` | 10 / 5 | Reuse v2's out-of-domain negatives. |
| `negative_near` | 8 / 4 | Real CS/ML topics you verified are **absent** from the corpus (grep the 1,000 titles and abstracts for the key term first). |

Write them in a spreadsheet or plain JSON as you go. Do not look at any
retrieval output while writing them.

**0d. Freeze the file.** `eval/questions-v3.json`:

```json
{
  "schema_version": 3,
  "frozen_at_utc": "...",
  "corpus_arxiv_ids_sha256": "7308d240...",
  "sampling_seed": 20260903,
  "retrieval_questions": [
    {"id": "v3q001", "split": "development", "type": "paraphrase",
     "query": "...", "relevant": {"2608.01301v2": 2}}
  ],
  "abstention_questions": [
    {"id": "v3n001", "split": "development", "type": "negative_near", "query": "..."}
  ]
}
```

`relevant` is a map from arXiv ID to grade (2 = primary, 1 = partial). For
paraphrase and lexical questions it has one entry with grade 2. For topical
questions leave it empty until 0e. Freeze timestamp goes in **after** 0e.

**0e. Pooling for topical questions.** Write `eval/tools/pool.py`:

- For each topical query run `retrieve` in every mode with `limit=20`.
- Union the arXiv IDs, shuffle with the seed, write
  `eval/tools/pool-<id>.md` with title + abstract, **no mode or rank shown**.
- Read each and assign 0/1/2. Put the grades into the question file and the
  full pool (including zeros) into a `pool` field so the verifier can check
  that every ranked ID in the evidence was in the judged pool.

Then set `frozen_at_utc` and never edit the file again.

**0f. Generalize the harness.** In `evaluation.py`:

- `MODES = ("vector", "keyword", "hybrid", "hybrid_rerank")`.
- `load_question_set` accepts schema 2 and 3; for schema 3 read `relevant` as
  a dict; for schema 2 wrap the list as `{id: 1 for id in list}`.
- `ndcg_at` takes graded gains: DCG uses `(2**grade - 1) / log2(rank+1)`.
- Add `recall_at(..., 10)` to each row and a `type` field copied from the
  question.
- `_aggregate` also groups by `type`. Output `aggregates[split][mode][type]`
  and `aggregates[split][mode]["all"]`.

In `evidence.py`, replace the hard-coded 90 / 20 / 10 with counts derived
from the question file (pass `--questions` to the retrieval verifier, which
the CLI already accepts as optional). Verify per-type aggregates too.

Tests to add in `tests/test_evaluation.py`: graded nDCG known answer
(`grades {a:2, b:1}` ranked `[b, a]` → compute by hand, assert), and a
schema-3 loader round trip on a tiny inline question set.

**0g. Run it.**

```bash
papertrail evaluate --questions eval/questions-v3.json \
  --manifest docs/evidence/corpus-manifest-1000.json \
  --output docs/evidence/phase-8-retrieval-v3-baseline.json
papertrail verify-evidence --kind retrieval --report docs/evidence/phase-8-retrieval-v3-baseline.json \
  --manifest docs/evidence/corpus-manifest-1000.json --questions eval/questions-v3.json
```

**Checkpoint 0:** per-type table in the notebook. Success condition: the
retrievers separate on dev nDCG@10 by type (no cell is a flat 1.00 for every
mode, and at least one type ranks the modes differently from another). Recall@5
is too coarse to be the gate with one relevant paper per query.

Two protocol rules, learned the hard way on 2026-09-05:

- Run the dry-run on the **development split only**. Never retrieve against
  held-out before the freeze. If held-out numbers were observed, replace the
  held-out questions with fresh ones (the v2 precedent) and do not dry-run them.
- Never edit a query *because* a specific retriever got it right or wrong.
  Edits before freezing must come from reading the query (wording, form,
  a wrong label), not from the dry-run table. Otherwise the benchmark is
  tuned against a retriever, which is the opposite of what it is for.

**Interview note:** "My first benchmark was title-derived and saturated. I
built one with paraphrase, lexical, and pooled topical queries so different
retrievers could fail in different places."

---

## Phase 1 — HNSW recall study (half a day)

**1a. Add an exact search path.** In `repository.py`, `vector_search` gets a
keyword arg `exact: bool = False`. When true, run
`session.execute(text("SET LOCAL enable_indexscan = off"))` before the select
(it only affects the current transaction).

**1b. Add `ef_search` control.** Same function: optional `ef_search: int |
None`; when set, `SET LOCAL hnsw.ef_search = :n`.

**1c. Write `eval/tools/hnsw_study.py`:**

```python
for q in all v3 retrieval queries (dev + heldout, both are fine: no tuning happens):
    exact = vector_search(session, emb, limit=50, exact=True)
    for ef in (40, 100, 200, 400):
        approx = vector_search(session, emb, limit=50, ef_search=ef)
        recall50 = |ids(approx) ∩ ids(exact)| / 50
        record (question_id, ef, recall50, latency_ms for both)
```

Write `docs/evidence/phase-8-hnsw-recall.json` with raw rows and a summary:
mean recall@50 per ef, p50/p95 latency per ef, and exact-scan p50/p95.

**1d. Optional index rebuild sweep.** `m` ∈ {16, 32}, `ef_construction` ∈
{64, 200}. Each is `DROP INDEX` + `CREATE INDEX ... WITH (m=.., ef_construction=..)`.
Restore the default at the end and re-run S3 to prove nothing changed.

**Checkpoint 1:** recall-vs-latency table. Expect recall@50 near 1.0 at every
ef for 2,039 vectors, and exact scan possibly *faster* than HNSW. Write that
down honestly; it is a good answer to "when would you not use an ANN index".

---

## Phase 1b — Abstention gate (1 day)

**1b-a. Collect gate signals.** Write `eval/tools/gate_signals.py`. For every
v3 retrieval question and every abstention question:

| Signal | How |
|---|---|
| `rrf_top` | current `hits[0]["score"]` in hybrid mode |
| `cos_top` | `vector_search(...)[0]["score"]` (already `1 - distance`) |
| `cos_margin` | `cos_top - vector_hits[1]["score"]` |
| `kw_top` | `keyword_search(...)[0]["score"]` or 0.0 if empty |
| `ce_top` | `hybrid_rerank` top `rerank_score` |
| `ce_margin` | top minus second `rerank_score` |

Save rows with `question_id, split, type, is_answerable, <signals>` to
`docs/evidence/phase-8-gate-signals.json`.

**1b-b. Score each signal.** Implement AUROC by hand in
`evaluation.py` (rank-sum formula; about 12 lines; add a known-answer test).
On the **development** split only, report AUROC per signal. Also pick the
balanced-accuracy threshold per signal with the existing `_choose_threshold`.

**1b-c. Report on held-out.** For the frozen threshold of each signal report:
false-refusal rate on answerable questions, false-answer rate on negatives,
and balanced accuracy. Put all signals side by side in one table. The current
`rrf_top` row is the baseline.

**1b-d. Switch the gate.** In `config.py` add `abstain_signal: str = "rrf_top"`
and keep `abstain_threshold`. In `generation.py`, compute the chosen signal
instead of reading `hits[0]["score"]` blindly. Default stays `rrf_top` until
the evidence says otherwise; then change the default **in a separate commit**
that cites the evidence file.

**1b-e. Soft abstention.** Add `AnswerResult.nearest_papers: list[...]` filled
on abstention with the top-3 retrieved (id, title, url). API and CLI print
them under "Low confidence; closest evidence:". No generation call happens.
Test: abstention response contains 3 nearest papers and no answer text.

**1b-f. Two-gate RAG evaluation.** Extend `evaluate-rag` with
`--verify-entailment`. Run held-out twice: current gate, and best-signal gate
+ entailment. Report answerable-answered, negatives-refused, entailment
refusals, and whether v3's equivalent of the two v2 false refusals now answer.

**Checkpoint 1b:** table of signals with AUROC and held-out error rates; the
default gate changed only if held-out improved without raising the
false-answer rate. Notebook entry explains the rank-quantization bug in your
own words with the 0.0328 / 0.0325 / 0.0318 numbers.

**Interview note:** "The confidence gate was reading rank positions, so a
top-1 vector hit got refused because the keyword ranker put it fifth. I
measured six candidate signals by AUROC and moved the gate to a continuous
score."

---

## Phase 2 — Reranker evaluation (half a day)

**2a. Pool-size sweep.** `retrieve` currently reranks `max(2*limit, 20)`
candidates. Make the pool size a parameter (`rerank_pool`), run v3 with
20 and 50.

**2b. Second model.** Add `bge-reranker-base` as a `CrossEncoderReranker`
model name (it is a sentence-transformers cross-encoder; loading is the same
call). Run both.

**2c. Dense-only rerank.** Add mode `vector_rerank` (rerank the vector list
alone). This separates "fusion helped" from "reranker helped".

**2d. Latency.** The harness already records per-query latency; report p50/p95
for each rerank configuration next to plain hybrid.

Evidence: `docs/evidence/phase-8-rerank.json`. Verifier: reuse the retrieval
verifier; add `reranker_model` and `rerank_pool` to `protocol` and check they
are present.

**Checkpoint 2:** table of (mode, model, pool) × (Recall@5, nDCG@10, p95).
The honest result may be "rerank adds 400 ms and +0.02 nDCG". Keep it.

---

## Phase 3 — Sparse retrieval (1 day)

**3a. Offline BM25 ablation (no DB writes).** `eval/tools/bm25_ablation.py`:

```bash
python -m pip install rank_bm25
```

- Load all chunks (`select id, arxiv_id, text from chunks`).
- Tokenize with the same regex as `keyword_search` plus lower-casing.
- `BM25Okapi(tokenized_chunks)`; for each v3 query take top-50 chunk IDs,
  collapse to distinct papers, compute Recall@5 / nDCG@10 per type.
- Compare with the `keyword` rows from the v3 baseline evidence file.

Decision rule: if BM25 beats Postgres FTS by more than 0.05 nDCG on `lexical`
dev questions, the ranking function is the problem and 3b-iii matters. If
not, the OR-query construction is the problem and 3b-i matters most.

**3b. Postgres-side changes**, each as its own commit with its own evidence run:

- **i. AND cascade.** In `keyword_search`, build `" & ".join(terms)` via
  `to_tsquery`; if it returns fewer than `limit` rows, union with the OR
  query results (AND hits first, then OR hits not already present). Keep the
  original OR behaviour reachable via a flag for A/B in the harness.
- **ii. Phrase boost.** If the query has 2+ consecutive capitalised tokens or
  a quoted span, add `phraseto_tsquery` hits with a rank bonus before the
  cascade.
- **iii. Field weights.** New Alembic migration replacing the generated
  column with `setweight(to_tsvector('english', title), 'A') ||
  setweight(to_tsvector('english', abstract_text), 'B')`. This needs the
  chunk text split back into title and body; the chunk already starts with
  the title followed by a blank line, so `split_part(text, E'\n\n', 1)` works.
  Then rank with `ts_rank_cd('{0.1,0.2,0.4,1.0}', search_vector, query, 32)`
  (the `32` normalizes by document length).
- **iv. Only if 3a says BM25 wins big:** evaluate ParadeDB `pg_search` in a
  second Compose service. Do not add it to the default stack unless it wins
  on held-out.

**Checkpoint 3:** keyword per-type table before/after each sub-step, plus the
BM25 offline row. Choose the configuration on dev; report held-out once.

**Interview note:** "OR-only matching made keyword search recall-heavy and
precision-poor; the AND cascade restored precision without the all-terms
failure I had already recorded in v1."

---

## Phase 4 — Fusion ablation (half a day)

**4a. Parameterize.** `reciprocal_rank_fusion` already takes `k`. Add
`weights: dict[str, float] | None` multiplying each source's `1/(k+rank)`.

**4b. Add a contrast fusion.** `convex_fusion(rankings, alpha)`: min-max
normalize each source's scores to [0,1], score = `alpha*vec + (1-alpha)*kw`.
This is what people usually do wrong; measure it rather than assume.

**4c. Sweep on dev only.** `k` ∈ {10, 30, 60, 100}; `w_vec` ∈ {1, 2, 3} with
`w_kw = 1`; `alpha` ∈ {0.5, 0.7, 0.9}; candidate pool 50 vs 200 per side.
Write `eval/tools/fusion_sweep.py` that calls `retrieve` with overrides and
produces one row per configuration per question.

**4d. Freeze one configuration** by dev nDCG@10 on `all`, report held-out
per type. The verifier must check `protocol.rrf_k` against the report's own
declared value, not the constant 60, and must reject any evidence file where
the chosen config was not the dev-best.

**Checkpoint 4:** per-type held-out comparison: v2-style RRF k=60 vs chosen.
Expected shape: hybrid wins `lexical`, loses a little on `paraphrase`.

---

## Phase 5 — Embedding model ablation (1 day)

**5a. Schema.** New migration adding `chunks.embedding_768 vector(768)` and a
second HNSW index. `embedding_runs` already records model and dimensions;
add a `column_name` field so a run knows where its vectors live.

**5b. Encoder changes.** `Encoder` gets an optional `query_prefix` /
`passage_prefix` (e5 needs `"query: "` and `"passage: "`; bge-small needs no
prefix for passages and an optional instruction for queries). Make the
prefixes part of the `embedding_runs` row so the mistake of forgetting them
is visible in evidence.

**5c. Embed and evaluate** each model: `bge-small-en-v1.5` (384),
`e5-small-v2` (384), `gte-small` (384), `bge-base-en-v1.5` (768).
`papertrail embed --model ... --column ...` then `papertrail evaluate --model ...`.
Record embedding wall time and index size (`pg_relation_size`).

**5d. Deliberate negative control.** Run e5 once *without* the prefixes and
keep the result. It should be worse. That row is the failure mode.

**Checkpoint 5:** table of model × per-type nDCG@10 × embed time × index size.
Do not change the default model unless held-out `all` improves and latency
does not regress; if you do, the corpus hash is unchanged but the evidence
files' `embedding_model` field must be updated everywhere it appears.

---

## Phases 6 and 7 — only if headroom remains

Follow the plan file. Both change either the corpus artefact (chunking) or
add nondeterminism (query rewriting); run them as ablations in
`eval/tools/`, never as default-path changes, unless a held-out gain justifies
the cost and you write the new claim boundary first.

---

## Wrap-up for each phase

1. Evidence file written, verifier passes, failure runs preserved.
2. README gains one table; `docs/results.md` gains the per-type breakdown.
3. Notebook entry with: the design choice, the failure mode, one number.
4. `candidate-profile.yaml` `learning_status` updated only after you can say
   items 3 out loud without notes.
5. Commit with the evidence path in the message. Do not push unless you decide
   to; the CLAUDE.md rule applies.

## Time budget

| Phase | Days |
|---|---|
| S + H | 0.75 |
| 0 | 2 |
| 1 | 0.5 |
| 1b | 1 |
| 2 | 0.5 |
| 3 | 1 |
| 4 | 0.5 |
| 5 | 1 |
| **Total through 5** | **~7.25** |

Stop after any checkpoint and you still have a complete, verifiable result.
