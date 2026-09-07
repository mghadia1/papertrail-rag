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

Exact scan: p50 23.1 ms, p95 34.5 ms (n=78). recall@50 below 1.0 at ef<50 is
truncation (HNSW returns at most ef rows), not approximation error, so it is only
meaningful from ef=100 up (0.998). Two findings at this corpus size: the default
ef=40 returns fewer rows (40) than the retrieval candidate pool asks for (50–200),
silently capping the vector side — a Phase 4 concern; and Postgres's planner picks
the HNSW index only at ef≤40 and reverts to an exact scan above that (`natural
scan` column). Latencies include per-call connection setup (a fresh session per
call), so read them only relative to each other within this file (A10); an
EXPLAIN ANALYZE execution-only comparison was exact ~10 ms vs index ~2 ms.

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

## Phase 1b D7 — two-gate RAG on held-out (September 6, 2026)

Held-out RAG generation with `openai/gpt-oss-120b` (Groq retired the previous
`llama-3.3-70b-versatile`). Three runs, each verified (`--kind rag`, 35 records):
`docs/evidence/phase-8-rag-{rrf_top,cos_mean_top3,cos_mean_top3-entail}.json`.

| gate (threshold) | answered/26 | refuse@rank1-2 | ood abstain | near abstain | entailment refusals | grounding |
|---|--:|--:|--:|--:|--:|--:|
| rrf_top (0.0324, current) | 13 (0.50) | 8 | 1.00 | 1.00 | 0 | 1.00 |
| cos_mean_top3 (0.4628) | 17 (0.65) | 1 | 0.78 | 0.50 | 0 | 1.00 |
| cos_mean_top3 + entailment | 0 (0.00) | 15 | 0.78 | 0.50 | 15 | n/a |

Reading: the current `rrf_top` gate leaves 13/26 answerable questions unanswered
and refuses 8 whose relevant paper was at hybrid rank 1-2; `cos_mean_top3` answers
17/26 and refuses only 1, but it also answers 2 of the 4 absent-topic (near-miss)
queries — the trade-off that keeps `rrf_top` as the default. Citation grounding is
1.00 under both gates. The statement-level NLI entailment gate at 0.80 refuses
every `gpt-oss-120b` answer (faithfulness 0.0-0.5) and needs recalibration for
this model before it is usable. `gpt-oss-120b` also omitted the required citation
format on 5-12 answers, which the citation gate correctly refused.

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

## Protocol history

The first report is retained because it showed keyword Recall@5 of 0.05 on
development queries. The OR-term keyword fix was made from that development
failure. A new v2 held-out set was then frozen. The first v2 RAG report is also
retained because two Groq 429 errors occurred; bounded retry fixed the operational
failure, and the reliable report was written separately.
