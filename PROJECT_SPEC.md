# PaperTrail — Research-Paper RAG & Semantic Search

> **Build spec, August 5, 2026.** One ML-focused project, designed to reach the caliber of
> Meet Sutariya's FitForge / TalentScope, while staying inside this portfolio's honesty rules
> and Mayank's real skills (Python, FastAPI, Docker, pytest, CI; Groq already used in
> `experiment-orchestrator`). Domain: **ML/CS research papers (arXiv)** — no medical, no
> robotics. Name is a placeholder — rename freely (PaperTrail, Citely, ScholarRAG...).

**Honesty gate (same as every project here):** nothing goes on the resume, the site, or an
application until Mayank can run it and explain, unaided, the retrieval design, one failure
mode, and why each metric was chosen. Real public data only. No claimed numbers until
measured. `resume_eligible: no` until that gate closes.

---

## 1. Why this project

Side by side with Meet's resume, the clearest gap is **production retrieval / RAG with
real ML underneath it**:

| Meet has | Mayank's current portfolio |
|---|---|
| Vector DB (pgvector), learned embeddings | none |
| Hybrid search (vector + full-text, Reciprocal Rank Fusion) | none |
| RAG pipeline + LLM generation with citations | none |
| Retrieval eval (MRR, Recall@k, ablations) | none |
| FastAPI + PostgreSQL + SQLAlchemy + Alembic | FastAPI only (resume-radar) |

`experiment-orchestrator` is agentic *tool-use*; `resume-radar` is a FastAPI scoring
service. Neither touches embeddings, a vector database, or RAG. PaperTrail fills that gap and
is squarely on-brand for an ML engineer: **semantic search and question-answering over real
arXiv ML/CS papers, with grounded citations.** Directly comparable to Meet's FitForge, but the
domain is machine-learning research — the field you're applying into.

## 2. What it does (one sentence)

Ask a technical question ("what's the difference between LoRA and full fine-tuning?");
PaperTrail retrieves the most relevant real arXiv papers using hybrid search, and an LLM
writes a short answer **with inline citations to the exact papers it used** — abstaining when
retrieval is too weak.

## 3. Where the real ML is

This is not just LLM-API glue — the ML content you can defend in an interview:

- **Learned embeddings.** Use a sentence-transformer to embed paper chunks; understand what
  the vectors mean, cosine similarity, and why semantic search beats keyword matching.
- **Retrieval as a ranking problem.** Reciprocal Rank Fusion, and evaluation with
  **MRR / Recall@k / nDCG** — the same rank-metric thinking used across ML retrieval and
  recommender systems.
- **A real evaluation harness with ablations** — your existing strength (SensorGuard,
  SurgiSeg) applied to retrieval: vector-only vs full-text vs hybrid, measured once on a
  held-out question set.
- **Optional modeling step:** fine-tune or train a small learned re-ranker (or cluster the
  embedding space with KMeans and inspect topics) to show model-building, not just API calls.

## 4. Architecture

```
        arXiv API / public metadata (real, free)
                     │  ingest N papers (title, abstract, authors, arXiv id)
                     ▼
        PostgreSQL  ── papers, chunks
           │  ├── pgvector column  (HNSW index)   ← semantic search (learned embeddings)
           │  └── tsvector column  (GIN index)    ← keyword search
           ▼
   Hybrid retriever: vector top-k  ⊕  full-text top-k
                     │  fused by Reciprocal Rank Fusion (k=60)
                     ▼
   RAG generator (Groq / openai gpt-oss-120b) → answer + inline [1][2] citations
                     │  abstains if fused top score < validation-tuned threshold
                     ▼
        FastAPI  /search  /ask  /health   (+ Docker, CI)
```

## 5. Tech stack (deliberately mirrors Meet's, all learnable in Python)

- **API:** FastAPI + Pydantic
- **Store:** PostgreSQL + **pgvector**, SQLAlchemy ORM, **Alembic** migrations
- **Embeddings:** `sentence-transformers` (e.g. `all-MiniLM-L6-v2`, CPU-friendly, free)
- **Keyword search:** PostgreSQL full-text (`tsvector` + GIN, `ts_rank_cd`)
- **Fusion:** Reciprocal Rank Fusion
- **Generation:** Groq free tier (`openai/gpt-oss-120b`; the earlier Llama 3.3 70B
  was retired by Groq) — $0.00
- **Data:** arXiv API (real, public) — or the public arXiv metadata snapshot on Kaggle
- **Ops:** Docker + docker-compose (app + Postgres), pytest, GitHub Actions CI
- **Stretch:** a learned re-ranker or KMeans topic clustering; Redis + background ingest
  worker; MCP server wrapper

New skills here: PostgreSQL/pgvector, embeddings, RAG, retrieval evaluation. Everything else
you already use.

## 6. Build order (each step ships something you can demo and explain)

1. **Ingest.** Pull ~1,000 real arXiv papers from a fixed set of categories (e.g. cs.LG,
   cs.CL, cs.CV). Store title/abstract/authors/arXiv id/date in Postgres. Chunk long
   abstracts. *Deliverable:* a populated DB + a row count you can quote.
2. **Vector search.** Embed chunks, add a pgvector HNSW index, expose `/search?mode=vector`.
   *Deliverable:* semantic search returning real arXiv IDs.
3. **Keyword + hybrid.** Add `tsvector`/GIN full-text, then fuse with Reciprocal Rank Fusion.
   `/search?mode=hybrid`. *Deliverable:* a query where hybrid clearly beats either alone —
   your interview story.
4. **RAG answer.** `/ask` — feed top-k fused chunks to Llama 3, require inline citations,
   **abstain** when the top fused score is below the tuned threshold. *Deliverable:* grounded
   answers + a refusal example.
5. **Evaluation harness (the part that impresses).** Hand-curate ~30 questions each with its
   known-relevant arXiv ID(s). Measure **Recall@5**, **MRR**, and **nDCG**. Run an
   **ablation:** vector-only vs full-text-only vs hybrid-RRF. Add a citation-grounding check
   (does every cited paper actually appear in the retrieved set?). *Deliverable:* one results
   table with real numbers.
6. **Harden.** Dockerize (compose app + Postgres), write tests (retrieval, RRF math, abstain
   logic, API), green GitHub Actions CI, `README`, `docs/how-it-works.md` — matching your
   other repos.
7. **Stretch / ML depth.** Train a small learned re-ranker (or cluster embeddings with KMeans
   and label topics) and measure whether it improves MRR; and/or expose `/search` and `/ask`
   as an **MCP server** tool. This connects to `experiment-orchestrator` and to Meet's MCP
   work (Portcullis).

## 7. The honest metrics you'll be able to claim (fill in once measured)

Do **not** put these on the resume until step 5 has produced them from a real run:

- Corpus size (real arXiv papers + IDs) and chunk count.
- Recall@5, MRR, nDCG on the hand-curated eval set — hybrid vs vector-only vs full-text-only.
- p50/p95 latency for vector and hybrid search.
- Abstain behavior: how many low-confidence questions were correctly refused.
- Citation grounding: % of answers whose citations all trace to retrieved papers.
- (If stretch) re-ranker's measured effect on MRR — including if it *didn't* help.

## 8. Resume bullets — TEMPLATE (bracketed values filled after measuring)

> Built **PaperTrail**, a research-paper RAG service over **[N] real arXiv ML/CS papers**:
> FastAPI + PostgreSQL/**pgvector** with an HNSW index over learned sentence-transformer
> embeddings, fusing semantic and full-text search via **Reciprocal Rank Fusion** — hybrid
> reached **[R@5] Recall@5 / [MRR] MRR** on a 30-question hand-curated eval set vs **[x]** for
> vector-only.

> Added grounded generation with **Groq/Llama 3** that cites the exact papers used and
> **abstains** below a validation-tuned retrieval threshold; **[G]%** of answers had every
> citation traceable to a retrieved source. Dockerized, **[T]** tests green in GitHub Actions.

## 9. Guardrails specific to this project

- Real arXiv data only; store arXiv IDs so every retrieval is verifiable.
- Tune the abstain threshold on a validation split, evaluate once on a held-out set — the
  same discipline SurgiSeg/SensorGuard already use.
- If you add the learned re-ranker, report its effect honestly even if it's zero or negative.
- `resume_eligible: no` until the explanation gate closes.

## 10. First session checklist (when you're ready to build)

1. `docker-compose` with Postgres + pgvector image; confirm `CREATE EXTENSION vector;`.
2. One Python script that pulls 20 arXiv papers and prints them — prove the data path.
3. Then follow build order 1→6. Ask me to scaffold any step.
