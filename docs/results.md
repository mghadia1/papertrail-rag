# Measured results — August 5, 2026

## Retrieval ablation

| Split / mode | Recall@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| Development vector | 1.00 | 1.00 | 1.00 | 22.03 | 30.45 |
| Development keyword | 0.90 | 0.906 | 0.916 | 24.25 | 39.48 |
| Development hybrid | 1.00 | 0.942 | 0.957 | 40.91 | 58.25 |
| Held-out vector | 1.00 | 1.00 | 1.00 | 24.16 | 44.78 |
| Held-out keyword | 0.90 | 0.85 | 0.886 | 27.57 | 46.19 |
| Held-out hybrid | 1.00 | 0.95 | 0.963 | 51.72 | 63.97 |

Vector was best on this controlled set. Hybrid preserved held-out Recall@5 but
moved `2608.01193v1` from rank 1 to rank 2. The result argues against assuming
fusion is automatically superior.

## Abstention and generation

- Development-selected threshold: `0.03239446668849102`.
- Development: 17/20 positive queries accepted; 10/10 negatives abstained.
- Held-out: 8/10 positive queries accepted; 5/5 negatives abstained.
- Reliable RAG run: 8 answers, 2 positive abstentions, 5 negative abstentions,
  zero provider/enforcement errors.
- All 8 generated answers cited only IDs from their retrieved set.

The 2 positive abstentions are false refusals under this label scheme. The
grounding result verifies citation membership, not semantic entailment.

## v3 graded evaluation (September 5, 2026)

The v2 set was saturated (every vector cell 1.00). v3 replaces it with graded
relevance over four question types — paraphrase (no title word appears in the
query), lexical (natural sentences carrying rare terms), pooled topical, and two
negative classes (out-of-domain and near-miss). 78 retrieval + 27 abstention
questions; four modes including `hybrid_rerank`. Evidence:
`docs/evidence/phase-8-retrieval-v3-baseline.json` (verified, 312 rows).

nDCG@10 by type (graded gains `2**grade - 1`):

| split · type | vector | keyword | hybrid | hybrid_rerank |
|---|---:|---:|---:|---:|
| dev · all | 0.876 | 0.759 | 0.879 | 0.942 |
| dev · paraphrase | 0.883 | 0.695 | 0.890 | 0.964 |
| dev · lexical | 1.000 | 0.977 | 1.000 | 1.000 |
| dev · topical | 0.699 | 0.595 | 0.697 | 0.821 |
| heldout · all | 0.894 | 0.762 | 0.913 | 0.951 |
| heldout · paraphrase | 0.883 | 0.735 | 0.969 | 1.000 |
| heldout · lexical | 1.000 | 1.000 | 1.000 | 1.000 |
| heldout · topical | 0.776 | 0.497 | 0.683 | 0.789 |

The types separate the retrievers: keyword collapses on paraphrase (no shared
title vocabulary), and on **the 6 held-out topical questions, RRF hybrid (0.683)
scored below plain vector (0.776)** — fusion hurt on that small set.
`hybrid_rerank` led every cell.
Abstention balanced accuracy fell to 0.828 dev / 0.846 held-out (from v2's
0.925 / 0.90) because the near-miss negatives — real ML topics verified absent
from the corpus — are harder to refuse than out-of-domain ones.

Latency note: the `hybrid_rerank` latency stored in this frozen file (dev p50
≈3896 ms) was polluted by cross-encoder model loading inside the timed loop and is
not a usable figure; it is left unedited (A3). The clean, warm-model measurement
is in the Phase 2 section below (dev p50 ≈170 ms). The nDCG@10 / Recall@10 values
here are deterministic and unaffected.

Topical grades were produced by two independent Claude gradings (Opus 4.8 draft,
Fable 5.1 blind), Cohen's kappa **0.719** (0.709 on the fully-blind pools); the
draft model (Opus 4.8) then adjudicated all 53 disagreements per abstract
(`eval/tools/adjudication.md`). This is a model-vs-model second opinion, not human
inter-annotator agreement, and no human has reviewed the labels yet. The held-out
split was authored from a sample disjoint from development and was not dry-run
before freeze.

## Phase 1 — HNSW index recall (September 5, 2026)

Chunk-level recall of the pgvector HNSW index (m=16, ef_construction=64,
vector_cosine_ops) against an exact sequential scan, over all 78 v3 retrieval
query vectors. The index is forced on (`SET LOCAL enable_seqscan = off`) so the
measurement is of the index, not the planner. Evidence:
`docs/evidence/phase-8-hnsw-recall.json` (verified, 467 rows).

| ef_search | recall@10 | recall@50 | rows | forced-idx p50 ms | natural scan |
|---|--:|--:|--:|--:|---|
| 10 | 0.957 | 0.200 | 10 | 3.6 | index |
| 40 (default) | 0.990 | 0.800 | 40 | 4.2 | index |
| 100 | 0.999 | 0.998 | 50 | 4.6 | seqscan |
| 200 | 1.000 | 1.000 | 50 | 4.9 | seqscan |
| 400 | 1.000 | 1.000 | 50 | 5.5 | seqscan |
| 1000 | 1.000 | 1.000 | 50 | 6.7 | seqscan |

Exact scan: p50 23.1 ms, p95 34.5 ms (n=78). recall@50 is undefined below ef=50
(HNSW returns at most ef rows, so the denominator cannot be reached); the sub-1.0
values at ef<50 are truncation, not approximation error, and the metric is only
meaningful from ef=100 up (0.998). Two findings at this corpus size: the default
ef=40 returns fewer rows (40) than the retrieval candidate pool asks for (50–200),
silently capping the vector side — a Phase 4 concern; and Postgres's planner picks
the HNSW index only at ef≤40 and reverts to an exact scan above that (`natural
scan` column). The table latencies include per-call connection setup (a fresh
session per call), so read them only relative to each other within this file
(A10); with that setup excluded, an EXPLAIN ANALYZE execution-only comparison was
exact ~10 ms vs index ~2 ms (versus the table's exact p50 of 23.1 ms, which
includes setup).

## Phase 1b — abstention gate signals (September 6, 2026)

Eight candidate confidence signals over all 105 v3 questions (78 answerable, 27
negatives). Signals selected on development by AUROC; held-out reported once with
the frozen development threshold. Evidence:
`docs/evidence/phase-8-gate-signals.json` and
`docs/evidence/phase-8-gate-selection.json` (verified, `--kind gate`). Held-out
n: 26 answerable, 5 out-of-domain, 4 near-miss.

| signal | dev AUROC | HO false-refuse /26 | HO false-answer near /4 | HO bal-acc |
|---|--:|--:|--:|--:|
| rrf_top (current gate) | 0.876 | 0.308 (8) | 0.000 (0) | 0.846 |
| cos_top | 0.993 | 0.077 (2) | 0.500 (2) | 0.850 |
| cos_mean_top3 (chosen) | 0.998 | 0.038 (1) | 0.500 (2) | 0.870 |
| kw_top | 0.947 | 0.231 (6) | 0.000 (0) | 0.885 |
| ce_margin | 0.811 | 0.154 (4) | 0.000 (0) | 0.923 |

The current `rrf_top` gate refuses **8 of 26** answerable held-out questions whose
relevant paper sat at hybrid rank 1 or 2 — a rank-quantization artifact, not
missing evidence. The AUROC-chosen `cos_mean_top3` removes almost all false
refusals and raises balanced accuracy, but it answers 2 of the 4 near-miss
negatives, so it is not adopted without review (brief rule D8). No default was
changed in this phase.

Selection-rule limitation: signals were chosen on development AUROC alone.
`ce_margin` had the best held-out balanced accuracy (0.923) with zero near-miss
false-answers but the lowest dev AUROC (0.811), so AUROC-only selection did not
pick it; a rule that priced in near-miss cost would have chosen differently. This
is a limitation of the AUROC-only criterion recorded here, not a new choice — the
held-out numbers were read once and no default was changed.

## Phase 1b D7 — two-gate RAG on held-out (September 6, 2026)

Held-out RAG generation with `openai/gpt-oss-120b` (Groq retired the previous
`llama-3.3-70b-versatile`). Three runs, each verified (`--kind rag`, 35 records):
`docs/evidence/phase-8-rag-{rrf_top,cos_mean_top3,cos_mean_top3-entail}.json`.

The three answerable outcomes are disjoint and sum to 26: answered, refused, and
uncited (the model omitted the `[id]` citation format, so citation enforcement
raised and the question was neither answered nor abstained). Rank-1/2 refusals are
split by cause — the gate vs the faithfulness heuristic — because they are
different failures.

| gate (threshold) | answered/26 | uncited/26 | gate refuse@1-2 | heuristic refuse@1-2 | ood abstain | near abstain | grounding |
|---|--:|--:|--:|--:|--:|--:|--:|
| rrf_top (0.0324, current) | 13 (0.50) | 5 | 8 | 0 | 1.00 | 1.00 | 1.00 |
| cos_mean_top3 (0.4628) | 17 (0.65) | 8 | 1 | 0 | 0.78 | 0.50 | 1.00 |
| cos_mean_top3 + heuristic faithfulness | 0 (0.00) | 10 | 1 | 14 | 0.78 | 0.50 | n/a |

Reading: the current `rrf_top` gate answers 13/26, refuses 8 whose relevant paper
was at hybrid rank 1-2, and 5 raised the no-citation error; `cos_mean_top3`
answers 17/26 and gate-refuses only 1, but it also answers 2 of the 4 absent-topic
(near-miss) queries — the trade-off that keeps `rrf_top` as the default. Citation
grounding is 1.00 among emitted answers under both gates.

The third row's faithfulness stage is a **token-overlap heuristic, not a trained
NLI model** — the only judge that exists in the package (see
`src/papertrail/entailment.py`, `HeuristicOverlapJudge`). At threshold 0.80 that
heuristic refuses every one of the 15 answers that passed citation enforcement
(faithfulness 0.0-0.5), so a real NLI judge — which was never built — would be the
prerequisite before this gate is usable. The three D7 files predate the
`entailment_judge` protocol field, so they do not record the judge; it was the
heuristic, the only one available. Finally, `gpt-oss-120b` omitting the citation
format on 5, 8, and 10 answerable questions is a **regression from the Llama 3.3
run** (which had zero enforcement errors), not working-as-designed behaviour; the
citation gate did its job, but the generation prompt needs model-specific citation
tuning (follow-up, measured on development only).

## Phase 2 (Part E) — reranker evaluation (September 6, 2026)

Pool-size × model × rerank-target study of the cross-encoder stage, over the 78
v3 retrieval questions (dev: 24 paraphrase, 16 lexical, 12 topical; held-out: 12
paraphrase, 8 lexical, 6 topical). Eight verified files,
`docs/evidence/phase-8-rerank-{msmarco,bge}-pool{20,50}-{hybrid_rerank,vector_rerank}.json`.
`hybrid_rerank` reranks the fused (RRF) list; `vector_rerank` reranks the vector
list alone. Latencies are within-file only (A10); each config ran in its own
process with the model load paid in a discarded warm-up.

nDCG@10 by type (lexical is 1.000 in every cell and omitted; Recall@10 is 1.000 in
every cell — reranking reorders a complete top-10, it does not add recall):

| config | dev all | dev para | dev top | HO all | HO para | HO top | dev p50 ms | dev p95 ms |
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

`*` topical lower bound: at pool 50 the first stage reaches outside the depth-20
judged pool, so 5-9 top-10 papers are unjudged and scored grade 0.

Findings:
- Reranking earns its place: every config beats plain hybrid on dev·all and
  dev·topical; the lift is entirely in topical and paraphrase (lexical is
  saturated).
- Fusion *and* the reranker both contribute. `vector_rerank` alone beats plain
  hybrid (dev·all 0.924 vs 0.879), but `hybrid_rerank` beats `vector_rerank` on
  dev·topical at every model/pool (e.g. ms-marco pool20 0.821 vs 0.773) — RRF
  fusion supplies topical signal the reranker cannot recover from vector alone.
- ms-marco vs bge: bge is marginally better (dev·all +0.009 at pool 20) for ~10×
  the latency (dev p50 1734 ms vs 170 ms).
- pool 20 vs pool 50: no reliable gain (ms-marco pool50 is worse on dev·all and
  held-out topical; pool-50 topical is a lower bound) for ~2× latency.

Recommendation: **keep the current default — ms-marco-MiniLM-L-6-v2,
`hybrid_rerank`, pool 20.** It captures ~all the quality lift (+0.063 dev·all over
plain hybrid) at ~110 ms added p50 latency; bge and pool 50 do not justify their
cost. No default was changed (Part E is a measurement phase). The frozen baseline's
`hybrid_rerank` latency (dev p50 3896 ms) was model-load-polluted; the clean
Phase-2 figure is 170 ms.

## Phase 3 (Part F) — sparse retrieval, F1: offline BM25 ablation (September 8, 2026)

Keyword is the weakest retriever in the v3 baseline. F1 asks whether that is the
*ranking function* or the *query construction*, by scoring the same corpus with
BM25 offline. `rank_bm25 0.2.2`, `BM25Okapi(k1=1.5, b=0.75)` over all 2,039 chunk
texts, tokenized with `keyword_search`'s exact regex, top-200 chunks →
`distinct_papers` → 10. **Development split only** (24 paraphrase, 16 lexical, 12
topical); held-out is reserved for one final report on the best configuration
(A1), and the verifier rejects a held-out row in this file. Evidence:
`docs/evidence/phase-8-bm25-offline.json` (verified, `--kind bm25`, 52 rows).

| type (dev) | FTS-OR `ts_rank_cd` | BM25 offline | gap |
|---|--:|--:|--:|
| all | 0.759 | 0.887 | +0.128 |
| lexical | 0.977 | 1.000 | +0.023 |
| paraphrase | 0.695 | 0.917 | **+0.222** |
| topical | 0.595 | 0.675* | +0.080* |

nDCG@10. Recall@10 also moves: all 0.923 → 0.981, paraphrase 0.833 → 0.958.

`*` **topical is a lower bound, not a clean comparison.** BM25 did not help build
the frozen topical judgment pools, so it surfaces papers nobody judged — all 12
topical rows do, 2–7 unjudged ids each in the top 10 — and those are scored grade
0. The pre-registered decision rule reads only `lexical` and `paraphrase`, whose
relevance is a fixed known-item set with no pool, so the branch below is unaffected
by this bias.

The decision rule was written into `docs/lab-notes.md` before the ablation ran: a
≥0.05 nDCG@10 gap on development lexical or paraphrase means the ranking function
matters. Paraphrase came in at **+0.222**, so the ranking function is the problem.

This is a **ranking failure, not a matching failure** — checked rather than
assumed. On the two worst paraphrase questions BM25 scores 1.000 and FTS 0.000,
yet FTS *did* match the relevant paper both times and merely ranked it 14th, just
outside the cutoff (v3q026: 2608.03291v1, matched set 639 papers; v3q004:
2608.01085v1, matched set 880 papers). `ts_rank_cd` scores from within-document
term frequency and cover density only — it carries no corpus-wide IDF term — and
at the default normalization flag it does not divide by document length, so on a
21–25-term paraphrase the common words count as much as the rare discriminative
ones. BM25 has both IDF and length normalization.

Caveat carried in the evidence file: Postgres `english` FTS stems and this offline
BM25 does not, so it compares ranking functions under different tokenization, not
a single controlled variable. BM25's per-query time (dev p50 2.9 ms) is in-process
scoring over a ~0.6 s in-memory index and is **not** comparable to the SQL path
(A10); nothing here proposes BM25 as a served retriever.

## Phase 3 (Part F) — F2-iii: field weights and length normalization (September 8, 2026)

F1 said the ranking function was the problem, so F2-iii adds the two levers
Postgres offers: a field-weighted `search_vector_weighted` column (title `A`, body
`B`, migration `20260908_0003`) ranked with
`ts_rank_cd('{0.1,0.2,0.4,1.0}', …, N)`. Development split only. Evidence:
`docs/evidence/phase-8-keyword-{or,or-depth200,weighted-n0,weighted-n1,weighted-n2}.json`
(verified, `--kind sparse`, 52 rows each).

nDCG@10, development:

| variant | all | lexical | paraphrase | topical |
|---|--:|--:|--:|--:|
| `or` (frozen baseline) | 0.759 | 0.977 | 0.695 | 0.595 |
| `or-depth200` | 0.759 | 0.977 | 0.695 | 0.595 |
| `weighted-n0` (weights only) | 0.721 | **1.000** | 0.602 | 0.588 |
| `weighted-n1` (÷ 1+log len) | 0.729 | **1.000** | 0.623 | 0.580 |
| `weighted-n2` (÷ len) | 0.480 | 0.923 | 0.314 | 0.222 |
| BM25 offline (F1) | 0.887 | 1.000 | 0.917 | 0.675 |

Two controls first. `or` **reproduces the frozen v3 baseline exactly**, so the
additive migration left the current default bit-identical and the harness is
validated against frozen evidence. `or-depth200` is identical to `or`, so the
candidate-depth asymmetry in F1 (BM25 read 200 candidate chunks, the SQL path 100)
is worth zero nDCG — that confound is closed, not assumed away.

**The result is negative: field weighting helps `lexical` and hurts `paraphrase`,
and the net is worse than the baseline** (0.759 → 0.721). Length normalization does
not rescue it, and dividing by document length is catastrophic (0.480).

The reason is measured, not guessed. v3 paraphrase queries are built so no title
word appears in the query: on development, a relevant paper's title shares a mean
of **0.04** content words with its paraphrase query (**23 of 24 share none**),
versus **4.81** for lexical queries (none has zero overlap). Weighting the title to
1.0 therefore boosts exactly the field a paraphrase query cannot match, and dilutes
the body evidence it can — while lexical rises to a perfect 1.000 for the same
reason in reverse.

**Conclusion: the missing ingredient is IDF, and Postgres's rank knobs cannot
supply it.** `ts_rank_cd` scores from within-document term frequency and cover
density and carries no corpus-wide document-frequency term; field weights and
length normalization are the only levers available, and both are
neutral-to-harmful here. No default changed — `keyword_search` still ranks the
unweighted column unless the new `weighted=`/`normalization=` arguments are passed.

Two corrections to the execution brief were needed and are recorded in
`docs/lab-notes.md`: `ts_rank_cd(..., 32)` divides the rank by itself+1, not by
document length, and is provably order-preserving (verified: flags 0 and 32 give an
identical top-15); and the migration is additive rather than replacing, so the
frozen keyword rows stay reproducible.

## Protocol history

The first report is retained because it showed keyword Recall@5 of 0.05 on
development queries. The OR-term keyword fix was made from that development
failure. A new v2 held-out set was then frozen. The first v2 RAG report is also
retained because two Groq 429 errors occurred; bounded retry fixed the operational
failure, and the reliable report was written separately.
