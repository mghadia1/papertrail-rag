# Project status

Date: August 5, 2026

- E1–E6: complete and locally verified.
- E7: complete; public repository and GitHub Actions are verified.
- E8: learning material complete; Mayank's unaided explanation check is pending.
- Resume eligible: **no** until E8 is passed.
- Corpus: 1,000 exact versioned arXiv IDs, 2,039 chunks, 2,039 normalized
  MiniLM vectors.
- Retrieval: vector, keyword, and deterministic `k=60` hybrid RRF through CLI
  and FastAPI.
- Generation: Groq `openai/gpt-oss-120b` (Groq retired the earlier Llama 3.3 70B
  on/before 2026-09; see docs/lab-notes.md), retrieved-ID citation enforcement,
  validation-selected abstention, and bounded transient-error retry.
- Held-out known-item retrieval: vector Recall@5/MRR/nDCG 1.00/1.00/1.00;
  hybrid 1.00/0.95/0.963; keyword 0.90/0.85/0.886.
- Held-out abstention: 8/10 answerable accepted and 5/5 out-of-domain refused.
- Reliable RAG run: 8 answers, zero provider/enforcement errors, all emitted
  citations present in retrieved sets.
- Verification: 49 local tests; evidence verifier passes all 90 retrieval rows
  and 15 RAG records; CPU-only Docker image builds.

A `hybrid_rerank` mode and an optional statement-level faithfulness check exist in
code. The faithfulness check is a token-overlap heuristic (`HeuristicOverlapJudge`),
not a trained NLI model — no NLI model exists in the package — and is off by
default. (`hybrid_rerank` is now measured in the v3 and reranker studies; see
docs/results.md.)

Claim boundary: title-derived known-item questions are not exhaustive relevance
judgments or production traffic. Citation set membership is not statement-level
entailment, and the optional faithfulness check is a lexical overlap heuristic,
not semantic entailment. PaperTrail remains off the résumé until the explanation
gate closes.
