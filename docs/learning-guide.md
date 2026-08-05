# PaperTrail learning guide

## The two retrieval signals

MiniLM converts text into 384 numbers. Similar directions in that vector space
often express similar meaning even without exact word overlap. Cosine similarity
compares direction. PostgreSQL full-text search instead rewards matching terms.
They fail differently, which is why fusion was worth testing.

HNSW is a graph index for approximate nearest-neighbor search. It can reduce
query work by navigating toward promising vectors, trading memory and possible
recall loss for speed. PaperTrail uses HNSW but did not measure exact-versus-HNSW
recall, so you should not claim that tradeoff was quantified.

RRF gives a result at rank `r` the score `1/(60+r)` from each list. A result
present in both lists receives both contributions. The constant 60 makes score
differences between nearby ranks less extreme. RRF avoids pretending cosine and
`ts_rank_cd` are on the same scale.

## How to explain the result

Say: “I expected hybrid search to help, but vector search already ranked every
labeled held-out paper first. Keyword noise moved one hybrid result to rank two,
so hybrid MRR fell to 0.95 and latency roughly doubled. I kept that result rather
than rewriting the project around the expected outcome.”

MRR focuses on the rank of the first relevant result. Recall@5 asks whether any
relevant result appears in the first five. nDCG rewards relevant results near the
top and can handle multiple relevance grades, although this set uses binary,
mostly single-paper labels.

## Abstention and grounding

The top hybrid RRF score is not a probability. The project selects a cutoff that
balances accepting development positives and refusing development negatives,
then freezes it. On held-out data it refused every negative but also refused two
labeled positives. That is the safety/coverage tradeoff.

Citation enforcement is code, not just prompting: an answer must contain a
versioned arXiv ID and every cited ID must be in the retrieved set. This prevents
invented references but cannot prove that the wording is fully supported.

## Run and inspect

```bash
papertrail search "multimodal agents environment distributions" --mode vector
papertrail search "multimodal agents environment distributions" --mode hybrid
papertrail ask "What does the retrieved work argue about environment design?"
papertrail verify-evidence --kind retrieval \
  --report docs/evidence/phase-6-retrieval-evaluation-v2.json \
  --manifest docs/evidence/corpus-manifest-1000.json
```

Open the raw evidence and locate `v2q30`, the rank-two hybrid failure. Then locate
`v2q28` and `v2q30` in the RAG report; both were below the frozen threshold.
