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

## Protocol history

The first report is retained because it showed keyword Recall@5 of 0.05 on
development queries. The OR-term keyword fix was made from that development
failure. A new v2 held-out set was then frozen. The first v2 RAG report is also
retained because two Groq 429 errors occurred; bounded retry fixed the operational
failure, and the reliable report was written separately.
