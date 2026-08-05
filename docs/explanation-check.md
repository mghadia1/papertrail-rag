# E8 explanation gate

Resume eligibility remains **no** until Mayank answers these aloud without
notes. A correct answer should use the measured project, not memorized slogans.

1. What does a 384-dimensional MiniLM embedding represent, and why use cosine
   similarity?
2. What does HNSW trade for speed, and what HNSW claim did this project *not*
   measure?
3. Compute the RRF contribution for rank 1 with `k=60`. Why not directly average
   cosine similarity and `ts_rank_cd`?
4. Which mode won held-out MRR, and why did hybrid rank `v2q30` worse?
5. Define Recall@5, MRR, and nDCG@10 in your own words.
6. How was the abstention threshold selected without using held-out labels?
7. Name the two answerable held-out questions that abstained. Is that good or
   bad, and what tradeoff does it show?
8. What exactly does “100% citation grounding” prove here, and what does it not
   prove?
9. Why are the v1 keyword failure and first 429-affected RAG run still checked
   in?
10. Give a two-minute architecture explanation from arXiv ingestion through
    `/ask`, then describe one retrieval failure and one provider failure.

When all ten answers are accurate without notes, record the date and reviewer
below. Only then may `resume_eligible` change to `yes`.

- Passed on: _not yet_
- Reviewed by: _not yet_
