# PaperTrail

PaperTrail is an evidence-first semantic-search and RAG service over a fixed
corpus of 1,000 real arXiv ML/CS papers. It combines normalized MiniLM
embeddings in PostgreSQL/pgvector, PostgreSQL full-text search, deterministic
Reciprocal Rank Fusion (RRF), and Groq generation with exact versioned-arXiv-ID
citations and confidence-based abstention.

**Resume status: not eligible.** The implementation and evaluation stages are
complete and the repository is public, but Mayank must still pass the unaided
explanation gate in [`docs/explanation-check.md`](docs/explanation-check.md).

## Verified system

- 1,000 versioned arXiv papers and 2,039 deterministic abstract chunks.
- Exact corpus ID SHA-256:
  `7308d240f717df7c9b17a1dfb7140c68615a462298b42866b95bc17f93250146`.
- 2,039 normalized 384-dimensional
  `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- pgvector cosine HNSW search, GIN-backed full-text search, and RRF with fixed
  `k=60`, exposed through CLI and FastAPI.
- `/ask` retrieves five hybrid results, abstains below the development-selected
  threshold `0.03239446668849102`, and rejects absent or invented citations.
- Groq retries 429 and transient 5xx responses with bounded backoff.
- 49 local tests, a CPU-only Docker image, migrations, Compose, and CI.
- Evaluation reports independently recompute from raw rows and fail if summary
  metrics, citations, thresholds, question IDs, corpus hash, or chronology are
  edited inconsistently.

An optional statement-level faithfulness check exists in code; it is a
**token-overlap heuristic (`HeuristicOverlapJudge`), not a trained NLI model** —
no NLI model exists in this package — and it is off by default with no headline
metric. (`hybrid_rerank` is now measured; see the reranker study below.)

## Frozen evaluation result

The evaluation uses 20 development and 10 held-out title-derived known-item
queries, plus 10 development and 5 held-out out-of-domain negatives. It tests
whether the known paper is retrieved; it is not an exhaustive topical relevance
study and does not represent production traffic.

| Held-out mode | Recall@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| Vector | 1.00 | 1.00 | 1.00 | 24.16 | 44.78 |
| Keyword | 0.90 | 0.85 | 0.886 | 27.57 | 46.19 |
| Hybrid RRF | 1.00 | 0.95 | 0.963 | 51.72 | 63.97 |

Vector retrieval won this set. Hybrid retained perfect Recall@5 but lowered MRR
because one relevant paper moved to rank 2, and it was slower because it executes
both retrieval paths. This result is more useful than an unsupported claim that
fusion must always improve ranking.

The abstention threshold was selected only from development scores. On the
untouched v2 held-out split it accepted 8/10 answerable queries and abstained on
5/5 out-of-domain queries (balanced accuracy 0.90). The reliable RAG run answered
those same 8 accepted queries, recorded zero provider/enforcement errors, and
had every emitted citation point to a retrieved paper. That 100% is a citation
set-membership check; it does **not** prove every generated statement is entailed
by its cited excerpt.

The first RAG run is intentionally preserved: two accepted questions hit Groq
HTTP 429 responses, producing 6/10 answers and two provider errors. After bounded
retry was added, the second separately named report completed with no errors.

## v3 graded evaluation

The v2 set above is saturated (every vector cell scores 1.00). v3 replaces it
with graded relevance over paraphrase, lexical, pooled-topical, and two negative
classes, evaluated across four modes including cross-encoder reranking. It is no
longer saturated — development vector nDCG@10 is 0.876 overall and 0.699 on
topical — and the retrievers separate by type: keyword collapses on paraphrase,
and on the 6 held-out topical questions RRF fusion (0.683) scored below plain
vector (0.776). Topical grades came from two independent model gradings (Claude
Opus 4.8 draft, Claude Fable 5.1 blind; Cohen's kappa 0.719); the draft model
then adjudicated every disagreement, recorded in `eval/tools/adjudication.md`. See
[`docs/results.md`](docs/results.md) for the per-type table and
[`docs/evidence/phase-8-retrieval-v3-baseline.json`](docs/evidence/phase-8-retrieval-v3-baseline.json)
(verified, 312 rows).

## HNSW index recall

Measured against an exact scan over the 2,039 vectors, the HNSW index is accurate
at the default `ef_search=40` (chunk-level recall@10 0.990; recall@50 is undefined
below ef=50 — the index returns at most ef rows — and reaches 0.998 once
`ef_search≥100`); see [`docs/results.md`](docs/results.md) and
[`docs/evidence/phase-8-hnsw-recall.json`](docs/evidence/phase-8-hnsw-recall.json)
(verified, 467 rows). At this corpus size Postgres's planner uses the index only
at the low default ef and otherwise runs an exact scan, so the ANN index buys
little here — an honest "not yet worth it at 2k vectors" result.

## Abstention-gate study

The confidence gate reads a rank-quantized RRF score, which refuses 8 of 26
held-out questions whose relevant paper was retrieved at rank 1-2; a continuous
`cos_mean_top3` gate cuts that to 1 but answers 2 of 4 absent-topic queries, so
the default stays `rrf_top` (the gate is now configurable). In the end-to-end RAG
run with `gpt-oss-120b`, that model omits the required `[id]` citation format on 5
(rrf_top) and 8 (cos_mean_top3) of 26 answerable questions — a regression from the
Llama run that the citation gate correctly refuses; every emitted answer's
citations were in its retrieved set (grounding 1.00). See
[`docs/results.md`](docs/results.md) and the verified
`docs/evidence/phase-8-gate-*.json` / `phase-8-rag-*.json`.

## Reranker study

A pool-size × model × target study of the cross-encoder stage (eight verified
files, `docs/evidence/phase-8-rerank-*.json`) confirms the current default —
`ms-marco-MiniLM-L-6-v2`, `hybrid_rerank`, pool 20 — is the right operating point:
it lifts development nDCG@10 from 0.879 (plain hybrid) to 0.942 at ~110 ms added
p50 latency, whereas `bge-reranker-base` adds only 0.009 for ~10× the latency and
a pool of 50 buys nothing reliable. Reranking the fused list beats reranking the
vector list alone on topical queries, so RRF fusion contributes signal the
reranker cannot recover on its own; Recall@10 stays 1.00 everywhere because
reranking reorders a complete top-10 rather than adding recall. No default was
changed; at pool 50 the first stage reaches outside the depth-20 judged pool, so
those topical scores are recorded lower bounds. See [`docs/results.md`](docs/results.md).

## Candidate-pool correctness fix

`retrieve()` asks Postgres for 50–200 vector candidates, but HNSW returns at most
`hnsw.ef_search` rows and that server default is 40 — so the vector side was capped
at 40 regardless of the pool size, including when 200 were requested. `retrieve()`
now sets `ef_search` to the pool size, which lifts the live vector candidate count
from 40 to 100 at the default `limit=10`. The abstention gate was re-checked first:
its accept/refuse decision flipped on zero of the 70 development questions, so the
frozen threshold is unchanged. Evidence:
[`docs/evidence/phase-8-fusion-heldout-v2.json`](docs/evidence/phase-8-fusion-heldout-v2.json).

## Sparse-retrieval study

Keyword is the weakest retriever, so Phase 3 asked whether that is the ranking
function or the query construction: an offline BM25 ablation beat Postgres FTS by
**+0.222 nDCG@10 on development paraphrase** queries, and on the two worst cases
FTS had *matched* the relevant paper and merely ranked it 14th — a ranking failure,
not a matching failure. Every Postgres-side fix was then measured and none earned
adoption: an AND-then-OR cascade gained +0.007 on development and **exactly +0.000
on held-out**, while a phrase boost and field weighting with length normalization
both made things worse (field weighting boosts the title, and 23 of 24 paraphrase
queries share no title word with their relevant paper). The gap is IDF, which
`ts_rank_cd` structurally lacks, so the keyword default is unchanged and a real
BM25 engine is left as a Phase 4 decision. See [`docs/results.md`](docs/results.md)
and the verified `docs/evidence/phase-8-bm25-offline.json` /
`phase-8-keyword-*.json`.

## Embedding model study

MiniLM-L6 was the only encoder ever tried, so Phase 5 embedded four alternatives into
their own columns and measured them. `bge-base-en-v1.5` (768-d) is genuinely the better
encoder — development vector nDCG@10 0.918 vs MiniLM's 0.876, and +0.018 on held-out —
but in the configuration PaperTrail actually serves, hybrid fusion, the held-out
difference is **−0.001**, for 2× the index size and ~2× the query-time latency, so the
default stays MiniLM. Two controls quantify the classic asymmetric-embedding mistake:
indexing e5 with `passage: ` and then querying without `query: ` scores 0.823, worse
than using no prefixes at all (0.833) and well below correct usage (0.852) — the CLI now
defaults the query prefix from the column's embedding run so it cannot happen by
accident. Topical scores for non-MiniLM encoders are recorded lower bounds, because the
frozen judgment pools were built from MiniLM-based retrievers. See
[`docs/results.md`](docs/results.md) and the verified `docs/evidence/phase-8-embed-*.json`.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev,ml]'
python -m pytest

docker compose up -d db
alembic upgrade head
papertrail verify-manifest docs/evidence/corpus-manifest-1000.json
papertrail search "robot learning from demonstration" --mode hybrid --limit 5
papertrail ask "How are surgical action triplets modeled?"

papertrail verify-evidence --kind retrieval \
  --report docs/evidence/phase-6-retrieval-evaluation-v2.json \
  --manifest docs/evidence/corpus-manifest-1000.json
papertrail verify-evidence --kind rag \
  --report docs/evidence/phase-6-rag-evaluation-v2-reliable.json \
  --manifest docs/evidence/corpus-manifest-1000.json \
  --questions eval/questions-v2.json
```

Set `GROQ_API_KEY` for generation. Search, evidence verification, and tests do
not need a Groq key. The checked-in `.env.example` contains no secret.

## Evidence

- [`docs/evidence/corpus-manifest-1000.json`](docs/evidence/corpus-manifest-1000.json)
- [`docs/evidence/phase-6-retrieval-evaluation.json`](docs/evidence/phase-6-retrieval-evaluation.json) — preserved v1 development failure
- [`docs/evidence/phase-6-retrieval-evaluation-v2.json`](docs/evidence/phase-6-retrieval-evaluation-v2.json) — frozen v2 result
- [`docs/evidence/phase-6-rag-evaluation-v2.json`](docs/evidence/phase-6-rag-evaluation-v2.json) — preserved 429 baseline
- [`docs/evidence/phase-6-rag-evaluation-v2-reliable.json`](docs/evidence/phase-6-rag-evaluation-v2-reliable.json) — retry-hardened run
- [`docs/evidence/phase-6-freeze-timestamp-correction.json`](docs/evidence/phase-6-freeze-timestamp-correction.json) — auditable metadata-only correction
- [`docs/evidence/phase-7-docker-api-smoke.json`](docs/evidence/phase-7-docker-api-smoke.json) — packaged health/search/ask smoke
- [`docs/evidence/phase-8-retrieval-v3-baseline.json`](docs/evidence/phase-8-retrieval-v3-baseline.json) — frozen v3 graded, per-type, four modes

Read [`docs/how-it-works.md`](docs/how-it-works.md),
[`docs/results.md`](docs/results.md), and [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for
the architecture, evaluation protocol, and claim boundaries.
