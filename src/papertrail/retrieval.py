"""Vector, keyword, and deterministic Reciprocal Rank Fusion retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from sqlalchemy.orm import Session

from .embedding import Encoder
from .repository import (
    keyword_search,
    require_vector_search_ready,
    vector_search,
)


SearchMode = Literal["vector", "keyword", "hybrid", "hybrid_rerank", "vector_rerank"]


def reciprocal_rank_fusion(
    rankings: dict[str, list[dict[str, object]]], *, k: int = 60
) -> list[dict[str, object]]:
    if k < 1:
        raise ValueError("RRF k must be positive")
    fused: dict[int, dict[str, object]] = {}
    for source in sorted(rankings):
        seen: set[int] = set()
        for rank, hit in enumerate(rankings[source], start=1):
            chunk_id = int(hit["chunk_id"])
            if chunk_id in seen:
                raise ValueError(f"ranking {source} contains duplicate chunk {chunk_id}")
            seen.add(chunk_id)
            if chunk_id not in fused:
                fused[chunk_id] = {
                    **hit,
                    "score": 0.0,
                    "component_ranks": {},
                }
            fused_hit = fused[chunk_id]
            fused_hit["score"] = float(fused_hit["score"]) + 1.0 / (k + rank)
            component_ranks = fused_hit["component_ranks"]
            assert isinstance(component_ranks, dict)
            component_ranks[source] = rank
    return sorted(
        fused.values(),
        key=lambda hit: (
            -float(hit["score"]),
            str(hit["arxiv_id"]),
            int(hit["chunk_id"]),
        ),
    )


def distinct_papers(
    hits: Iterable[dict[str, object]], *, limit: int
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for hit in hits:
        arxiv_id = str(hit["arxiv_id"])
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        selected.append(hit)
        if len(selected) == limit:
            break
    return selected


def _rerank(
    query: str,
    pooled: list[dict[str, object]],
    *,
    top_k: int,
    reranker: object | None,
) -> list[dict[str, object]]:
    """Apply a cross-encoder reranker to a first-stage candidate pool.

    Defaults to the real cross-encoder, which raises if the model cannot load
    rather than silently substituting the lexical reranker (brief A6). The actual
    pool size seen by the reranker is stamped on every returned row so the study
    can record it (brief E, trap: pool 50 may be smaller than requested).
    """
    from .reranking import CrossEncoderReranker

    active_reranker = reranker if reranker is not None else CrossEncoderReranker()
    reranked = active_reranker.rerank(query, pooled, top_k=top_k)
    pool_size = len(pooled)
    for row in reranked:
        row["rerank_pool_size"] = pool_size
    return reranked


def retrieve(
    session: Session,
    query: str,
    *,
    mode: SearchMode,
    limit: int,
    encoder: Encoder | None = None,
    rrf_k: int = 60,
    reranker: object | None = None,
    rerank_pool: int | None = None,
) -> list[dict[str, object]]:
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    if rerank_pool is not None and rerank_pool < limit:
        raise ValueError("rerank_pool must be at least limit")
    candidate_limit = min(200, max(50, limit * 10))
    if mode == "keyword":
        return distinct_papers(
            keyword_search(session, query, limit=candidate_limit), limit=limit
        )
    if encoder is None:
        raise ValueError(f"{mode} retrieval requires an encoder")
    require_vector_search_ready(
        session, model_name=encoder.model_name, dimensions=encoder.dimensions
    )
    query_embedding = encoder.encode([query], batch_size=1)[0]
    vector_hits = vector_search(session, query_embedding, limit=candidate_limit)
    if mode == "vector":
        return distinct_papers(vector_hits, limit=limit)

    # First-stage pool depth for a rerank stage; the default preserves the
    # original hybrid_rerank behaviour (max(limit*2, 20)).
    pool = rerank_pool if rerank_pool is not None else max(limit * 2, 20)

    if mode == "vector_rerank":
        # Rerank the vector list alone — no keyword_search call on this path.
        pooled = distinct_papers(vector_hits, limit=pool)
        return _rerank(query, pooled, top_k=limit, reranker=reranker)

    if mode not in {"hybrid", "hybrid_rerank"}:
        raise ValueError(f"unsupported search mode: {mode}")
    keyword_hits = keyword_search(session, query, limit=candidate_limit)
    fused = reciprocal_rank_fusion(
        {"keyword": keyword_hits, "vector": vector_hits}, k=rrf_k
    )
    if mode == "hybrid":
        return distinct_papers(fused, limit=limit)

    pooled = distinct_papers(fused, limit=pool)
    return _rerank(query, pooled, top_k=limit, reranker=reranker)

