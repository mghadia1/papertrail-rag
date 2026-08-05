# How PaperTrail works

## Data and provenance

PaperTrail fetches public arXiv Atom metadata, normalizes it, and creates
deterministic overlapping abstract chunks with the title prepended. A manifest
stores all 1,000 sorted, versioned IDs and their SHA-256. Replay fetches only
those IDs; verification compares complete ID sets rather than counts alone.

Each chunk stores a normalized 384-dimensional MiniLM vector and a generated
English `tsvector`. An `embedding_runs` row binds vectors to model, dimensions,
normalization, corpus hash, and coverage. Retrieval refuses incomplete or
incompatible provenance.

## Retrieval

- Vector mode encodes the query with the same MiniLM model and orders chunks by
  pgvector cosine distance using an HNSW index.
- Keyword mode extracts query terms, joins them with OR, converts them through
  `websearch_to_tsquery`, and ranks GIN-indexed matches with `ts_rank_cd`.
- Hybrid mode takes independent candidate lists and adds `1/(60 + rank)` per
  appearance. Deterministic tie-breaking and one-result-per-paper deduplication
  make output reproducible.

RRF combines ranks rather than adding incomparable cosine and full-text score
scales. It is not guaranteed to beat the strongest component: vector achieved
held-out MRR 1.00 here, while hybrid achieved 0.95.

## Grounded answering and abstention

`/ask` retrieves five hybrid chunks and checks the top RRF score before making
an LLM request. The threshold `0.03239446668849102` was selected from development
positive and out-of-domain scores, never the held-out split. Below it, the
system returns an abstention without generation.

Above it, the prompt treats excerpts as untrusted data, requires exact
versioned arXiv IDs, and limits the answer to supplied evidence. Code then
parses citations, rejects an answer with no citation, and rejects any ID absent
from the retrieved set. Groq 429 and transient 5xx failures receive at most four
attempts with bounded `Retry-After`/exponential backoff.

## Evaluation protocol

Version 1 exposed that an all-terms keyword query made keyword retrieval nearly
useless. That report was preserved. Only development observations informed the
OR-term fix. Version 2 reused the original development set but introduced a new
held-out set before rerunning evaluation.

The v2 set contains 30 title-derived known-item questions (20 development, 10
held-out) and 15 out-of-domain negatives (10/5). It reports Recall@5, MRR,
nDCG@10, and post-warm-up p50/p95 latency for all modes. The abstention threshold
is selected on development scores, then frozen for held-out retrieval and RAG.

Evidence verification recomputes aggregate metrics and rates from raw records.
It also validates the corpus hash, RRF constant, frozen question IDs, citation
parsing, grounding flags, threshold, and freeze-before-run chronology.

## Failure modes and limitations

- The questions are title-derived known-item lookups with one labeled paper;
  perfect vector Recall@5 does not imply perfect open-ended search.
- HNSW is approximate, but this study did not compare HNSW against exact vector
  search, so no index-recall claim is made.
- RRF score magnitude depends on candidate ranks and `k`; it is not a calibrated
  probability. The threshold rejected 2/10 answerable held-out questions.
- One relevant paper moved to rank 2 under hybrid fusion, showing keyword noise
  can hurt a strong vector ranker.
- Citation membership is weaker than sentence-level entailment or factuality.
- The corpus covers recent papers in three categories, not all ML/CS literature.
- arXiv and Groq are external services; replay uses polite pacing and Groq uses
  bounded retry, but persistent outages still surface as errors.
