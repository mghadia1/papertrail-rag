# PaperTrail retrieval upgrade plan

Date: 2026-09-02. Baseline: commit `88b1b67`, 48 tests passing, Docker DB not
running on this machine (every phase below except 0a and 3a needs `docker
compose up -d db`).

## Where the retrieval stack stands today

| Layer | Current implementation | Gap |
|---|---|---|
| Corpus | 1,000 arXiv papers, abstracts only, 2,039 chunks (1,000 chars, 150 overlap, title prepended) | ~2 chunks per paper; no body text |
| Dense | `all-MiniLM-L6-v2`, 384-d, normalized, pgvector HNSW cosine, default index params | No exact-kNN vs HNSW recall check; one embedding model only |
| Sparse | Postgres FTS: terms ≥3 chars joined with `OR`, `websearch_to_tsquery`, `ts_rank_cd` | Not BM25; OR-only means any common term matches; no phrase or AND fallback |
| Fusion | RRF, fixed `k=60`, equal weights, candidate pool = `min(200, max(50, 10*limit))` | `k` and weights never ablated |
| Rerank | `hybrid_rerank` mode: `CrossEncoderReranker` (`ms-marco-MiniLM-L-6-v2`) with a **silent lexical fallback** when torch is absent | Never evaluated; not in `evaluation.MODES`; not in README; fallback can masquerade as a cross-encoder |
| Gate | Top fused RRF score vs frozen threshold `0.03239` | Rank-quantized signal; refused 2/10 answerable held-out questions because the keyword list ranked the right paper lower |
| Eval | 30 title-derived known-item queries + 15 negatives; Recall@5 / MRR / nDCG@10 / latency | **Saturated**: vector scores 1.00 on every metric, so no retrieval change can register |

The last row is the whole problem. Nothing else in this plan is worth doing
before it is fixed, because every result would be "1.00 → 1.00".

## Honesty items to close first (no new features)

1. README still says 41 tests and does not mention `hybrid_rerank` or
   entailment. Either document them as *implemented, unevaluated* or leave them
   out of the verified-system list. Do not list them as results.
2. `CrossEncoderReranker._load_model` swallows every exception and falls back to
   lexical scoring while keeping `rerank_score` in the output. Any evidence file
   produced this way would be mislabeled. Fix: fallback only when explicitly
   requested (`allow_fallback=True`), otherwise raise; always record the
   reranker's real `model_name` in the evidence row.
3. `docs/status.md` and `PROJECT_STATUS.md` should say the reranker and NLI
   paths exist but carry no measured metric.

## Phase 0 — Build an evaluation that can move

Goal: a v3 question set where vector-only does **not** score 1.00, frozen
before any tuning, with the same dev/held-out discipline as v2.

- **0a. Query types (target ~60 dev / 30 held-out, plus 15/10 negatives):**
  - *Paraphrase known-item*: write the query from the abstract's method or
    result, never reusing title words. This is where MiniLM should start to miss.
  - *Lexical-precise*: acronyms, model names, dataset names, numbers
    ("SAM2", "Kvasir", "3.3 70B"). This is where dense should lose and sparse
    should win, which makes fusion measurable.
  - *Topical multi-relevant*: "papers that use diffusion for planning". Label by
    **pooling**: take top-20 from vector, keyword, hybrid, and rerank, judge each
    pooled paper relevant / not, record the pool so the verifier can recompute.
    Graded labels (2 = on topic, 1 = partial) make nDCG meaningful.
  - *Negatives*: keep the out-of-domain set, add near-domain negatives
    (real CS topics absent from the corpus) so abstention is tested harder.
- **0b. Freeze**: `eval/questions-v3.json` with `schema_version: 3`, pools,
  grades, `frozen_at_utc`, corpus hash. Extend `load_question_set` and
  `verify-evidence` for graded relevance and pooled judgments.
- **0c. Add metrics**: Recall@10, nDCG@10 with graded gains, and per-query-type
  breakdown so "hybrid helps lexical-precise, hurts paraphrase" is a reportable
  finding rather than one blended number.
- **0d. Wire `hybrid_rerank` into `evaluation.MODES`.**
- **Done when**: dev vector Recall@5 < 0.9 on at least one query type, held-out
  untouched, verifier passes.

Effort: 1–2 days, mostly labeling. This is the interview story: "my first eval
was saturated and I had to design one that could fail."

## Phase 1 — Index recall study (cheap, closes a stated gap)

`how-it-works.md` says no HNSW recall claim is made. Make one.

- Add an exact-scan path (`SET LOCAL enable_indexscan = off` or
  `ORDER BY embedding <=> q` with `hnsw.ef_search` disabled) and compute
  recall@50 of HNSW vs exact for every eval query.
- Sweep `hnsw.ef_search` ∈ {40, 100, 200, 400} and index `m` / `ef_construction`
  (needs a rebuild per setting; corpus is small enough).
- Report recall vs p50/p95 latency. At 2,039 vectors the honest finding may be
  "exact scan is faster than HNSW". Keep it if so.
- Design choice to defend: why cosine over an HNSW index on normalized vectors
  equals inner product. Failure mode: ef_search below k silently truncates.

Effort: half a day.

## Phase 1b — Redesign the abstention gate (the "I don't know" problem)

Observed on the v2 held-out RAG run: 2 of 10 answerable questions were refused.
The evidence file shows why. The gate compares the **RRF score of the top fused
hit** against `0.03239`, and RRF scores are quantized by rank position, not by
match quality:

| Top score | Meaning | Gate |
|---|---|---|
| 0.03279 | rank 1 in vector *and* keyword | answered |
| 0.03252 | rank 1 in one list, rank 2 in the other | answered |
| 0.03178 | rank 1 in vector, rank 5 in keyword (v2q30) | **refused** |
| 0.03175 | rank 3 in both lists (v2q28) | **refused** |
| 0.016–0.030 | out-of-domain negatives | refused |

The threshold sits between "top-2 in both lists" and everything else. A query
whose correct paper is the top vector hit gets refused because the keyword
ranker placed it fifth. The gate is measuring keyword noise, not evidence
quality, and the docs already admit RRF "is not a calibrated probability".

- **1b-a. Log candidate gate signals** for every dev/held-out query and every
  negative, in one evidence file: top vector cosine, cosine margin between
  hit 1 and hit 2, top keyword `ts_rank_cd`, RRF top score (current), and once
  Phase 2 runs, the cross-encoder score. Plot each signal's positive vs
  negative distributions and report AUROC per signal. This alone answers
  "which signal separates answerable from unanswerable".
- **1b-b. Pick the gate on dev only**, same balanced-accuracy procedure as
  `_choose_threshold`, but on the best-AUROC signal. Expected outcome: top
  vector cosine or cross-encoder score beats RRF, because they are continuous
  and do not depend on the other ranker's rank.
- **1b-c. Two-sided cost.** Report false-refusal rate on answerable questions
  and false-answer rate on negatives separately, not only balanced accuracy.
  Refusing a real question and answering a fake one are different failures;
  a small corpus makes the first one more common, and the write-up should
  say which one the threshold favours.
- **1b-d. Soft abstention instead of a flat refusal.** When the gate fails,
  return the nearest papers with a "low confidence, here is the closest
  evidence" response rather than nothing. With a 1,000-paper corpus most real
  questions are partly out of domain, so a graded answer is more useful than
  a binary one. Keep the citation-enforcement rules identical.
- **1b-e. Move the second gate after generation.** The NLI entailment check
  already exists in `generation.py`. Evaluate it on v3 as the post-answer
  gate: retrieval gate loose (catch obvious out-of-domain), entailment gate
  strict (catch ungrounded answers). Report how many of the 2/10 false
  refusals become correct answers under the new pair.
- **On the "small database" concern:** a 1,000-paper corpus *should* refuse a
  lot of arbitrary questions. That is correct behaviour, not a bug. The bug is
  refusing questions whose answer is in the corpus. Phase 1b fixes the latter;
  growing the corpus (which changes the hash and every frozen evidence file)
  is a separate decision and is not recommended before the gate is fixed.
- Design choice to defend: gating on a continuous similarity instead of a
  rank-fused score. Failure mode found: a rank-based score refusing a top-1
  vector hit because of the *other* ranker.

Effort: 1 day after Phase 0, and the highest-value fix in this plan for the
user-facing behaviour.

## Phase 2 — Evaluate the reranker that already exists

- Install `.[ml]`, confirm `CrossEncoderReranker` really loads (after the
  fallback fix), run v3 for `hybrid_rerank` with candidate pool sizes 20 / 50.
- Compare against `bge-reranker-base` as a second model.
- Record latency: a cross-encoder on 50 pairs on CPU will likely add 200–800 ms.
  That trade-off is the result, whichever way it goes.
- Also test "vector_rerank" (rerank the dense list only) so you can say whether
  the reranker or the fusion did the work.

Effort: half a day after Phase 0.

## Phase 3 — Fix the sparse side

The current keyword path is the weakest component and the easiest to explain.

- **3a. In-process BM25 ablation** (`rank_bm25`, offline, no DB): tokenise the
  2,039 chunks, score the v3 queries, compare against Postgres `ts_rank_cd`.
  Do this first because it tells you whether the FTS ranking function or the
  OR-query construction is the problem.
- **3b. Postgres-side improvements**, in order of expected gain:
  - Cascading query: try AND of all terms, fall back to OR only when AND
    returns fewer than `limit` rows. This restores precision without the v1
    "all terms or nothing" failure.
  - Phrase boost: `phraseto_tsquery` for quoted or bigram spans.
  - `ts_rank_cd` normalization flags (`32` for length norm) or `ts_rank` with
    weights: title in `A`, abstract in `B` via `setweight`. Requires a migration
    to rebuild `search_vector`.
  - Optional: ParadeDB `pg_search` for real BM25 inside Postgres. Only if 3a
    shows BM25 clearly beats FTS; otherwise the added extension is not earned.
- Design choice to defend: OR-then-rank vs AND cascade. Failure mode: stop-word
  heavy queries returning the entire corpus under OR.

Effort: 1 day.

## Phase 4 — Fusion ablation

Once sparse is credible, fusion becomes interesting.

- Sweep RRF `k` ∈ {10, 30, 60, 100}.
- Weighted RRF (`w_vec`, `w_kw`) and a min-max score convex combination as a
  contrast. Select on dev only, freeze, report held-out.
- Candidate pool size ablation (50 vs 200 per side).
- Per-query-type breakdown from Phase 0c is the deliverable: expect hybrid to
  win on lexical-precise queries and lose slightly on paraphrase.

Effort: half a day; mostly running the harness.

## Phase 5 — Embedding model ablation

`embedding_runs` already stores model, dimensions, and normalization, so a
second model is a schema-compatible new run.

- Candidates, all CPU-feasible at 2,039 chunks: `bge-small-en-v1.5` (384-d),
  `e5-small-v2` (needs `query:` / `passage:` prefixes), `gte-small`,
  `bge-base-en-v1.5` (768-d, needs a second column or table).
- Keep MiniLM as baseline. Report Recall / nDCG / embedding time / index size.
- Failure mode to look for: e5 without the prefix instruction scores worse
  than MiniLM, a real and common bug.

Effort: 1 day including a migration for a 768-d column.

## Phase 6 — Representation and chunking (lower priority)

- Title-prepend on/off, chunk size 500 / 1,000 / 1,500 with the same overlap.
- Separate title embedding fused as a third RRF ranking.
- Full-text ingestion is out of scope for now: it changes the corpus hash,
  every frozen evidence file, and the replay contract. Only revisit if the
  abstract-only ceiling becomes the finding.

## Phase 7 — Query-side rewriting (optional, nondeterministic)

HyDE or multi-query expansion through Groq. Costs a generation call per
search, breaks the "search needs no API key" property, and adds nondeterminism
to evidence. Do it last, only as an ablation, and keep it off the default path.

## Evidence and documentation for every phase

- New `docs/evidence/phase-8-<topic>.json` per study; failures preserved, never
  overwritten (same rule as the v1 keyword failure).
- Extend `verify-evidence` so any new metric recomputes from raw rows.
- README gets one new table per completed phase and the *per-query-type*
  breakdown; the current v2 table stays as the historical baseline.
- Update `candidate-profile.yaml` `learning_status` only when Mayank can give
  the design choice and failure mode for that phase unaided.

## Suggested order and why

| Order | Phase | Reason |
|---|---|---|
| 1 | Honesty items | Cheap, and everything after depends on evidence being labeled correctly |
| 2 | 0 Eval v3 | Nothing is measurable without it |
| 3 | 1 HNSW recall | Half a day, closes a documented gap, needs no new models |
| 4 | 1b Abstention gate | Directly fixes the observed 2/10 false refusals; the gate currently reads keyword rank noise |
| 5 | 2 Reranker | Code exists; measuring it is the missing half, and its score is a gate candidate |
| 6 | 3 Sparse | Biggest expected gain per hour; strongest interview story |
| 7 | 4 Fusion | Only meaningful after 3 |
| 8 | 5 Embeddings | Good story, more setup |
| 9 | 6, 7 | Only if the earlier phases leave headroom |

Roughly six working days through Phase 5. Stop and write up after any phase;
each one produces a standalone, verifiable result.
