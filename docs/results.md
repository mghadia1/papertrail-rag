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

## Protocol history

The first report is retained because it showed keyword Recall@5 of 0.05 on
development queries. The OR-term keyword fix was made from that development
failure. A new v2 held-out set was then frozen. The first v2 RAG report is also
retained because two Groq 429 errors occurred; bounded retry fixed the operational
failure, and the reliable report was written separately.
