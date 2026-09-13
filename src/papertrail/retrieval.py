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


def _fusion_sort(fused) -> list[dict[str, object]]:
    """Deterministic ordering shared by every fusion function."""
    return sorted(
        fused,
        key=lambda hit: (
            -float(hit["score"]),
            str(hit["arxiv_id"]),
            int(hit["chunk_id"]),
        ),
    )


def reciprocal_rank_fusion(
    rankings: dict[str, list[dict[str, object]]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Reciprocal Rank Fusion, optionally weighting each source.

    Each source contributes ``weights[source] * 1/(k + rank)``; an absent source
    weight defaults to 1.0, so ``weights=None`` is the original unweighted fusion.
    """
    if k < 1:
        raise ValueError("RRF k must be positive")
    fused: dict[int, dict[str, object]] = {}
    for source in sorted(rankings):
        weight = float((weights or {}).get(source, 1.0))
        if weight < 0:
            raise ValueError(f"fusion weight for {source} must be non-negative")
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
            fused_hit["score"] = float(fused_hit["score"]) + weight / (k + rank)
            component_ranks = fused_hit["component_ranks"]
            assert isinstance(component_ranks, dict)
            component_ranks[source] = rank
    return _fusion_sort(fused.values())


def _min_max(hits: list[dict[str, object]]) -> dict[int, float]:
    """Min-max normalize a source's own scores into [0, 1].

    Note the lowest-scoring candidate in a truncated list normalizes to exactly
    0.0 — expected, not a bug (brief G trap). A list whose scores are all equal
    normalizes to 1.0, since every member is jointly the best that source offers.
    """
    scores = [float(hit["score"]) for hit in hits]
    if not scores:
        return {}
    low, high = min(scores), max(scores)
    span = high - low
    return {
        int(hit["chunk_id"]): (1.0 if span == 0 else (float(hit["score"]) - low) / span)
        for hit in hits
    }


def convex_fusion(
    rankings: dict[str, list[dict[str, object]]], *, alpha: float
) -> list[dict[str, object]]:
    """Score-based fusion: ``alpha * vector + (1 - alpha) * keyword``.

    Each source's raw scores are min-max normalized within its own list first,
    because an RRF score, a cosine similarity and a ts_rank_cd value are not on a
    common scale. A document missing from a source contributes 0 for that source.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("convex fusion alpha must be within [0, 1]")
    unknown = set(rankings) - {"vector", "keyword"}
    if unknown:
        raise ValueError(f"convex fusion expects vector/keyword sources; got {sorted(unknown)}")
    weight_for = {"vector": alpha, "keyword": 1.0 - alpha}

    normalized = {source: _min_max(hits) for source, hits in rankings.items()}
    fused: dict[int, dict[str, object]] = {}
    for source in sorted(rankings):
        seen: set[int] = set()
        for rank, hit in enumerate(rankings[source], start=1):
            chunk_id = int(hit["chunk_id"])
            if chunk_id in seen:
                raise ValueError(f"ranking {source} contains duplicate chunk {chunk_id}")
            seen.add(chunk_id)
            if chunk_id not in fused:
                fused[chunk_id] = {**hit, "score": 0.0, "component_ranks": {}}
            fused_hit = fused[chunk_id]
            fused_hit["score"] = float(fused_hit["score"]) + weight_for[source] * normalized[source][chunk_id]
            component_ranks = fused_hit["component_ranks"]
            assert isinstance(component_ranks, dict)
            component_ranks[source] = rank
    return _fusion_sort(fused.values())


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
    weights: dict[str, float] | None = None,
    fusion: str = "rrf",
    alpha: float = 0.5,
    candidate_limit: int | None = None,
    ef_search: int | None = None,
    keyword_strategy: str = "or",
) -> list[dict[str, object]]:
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    if rerank_pool is not None and rerank_pool < limit:
        raise ValueError("rerank_pool must be at least limit")
    if fusion not in {"rrf", "convex"}:
        raise ValueError(f"unsupported fusion: {fusion!r}")
    if candidate_limit is None:
        candidate_limit = min(200, max(50, limit * 10))
    elif candidate_limit < limit:
        raise ValueError("candidate_limit must be at least limit")
    if mode == "keyword":
        return distinct_papers(
            keyword_search(
                session, query, limit=candidate_limit, strategy=keyword_strategy
            ),
            limit=limit,
        )
    if encoder is None:
        raise ValueError(f"{mode} retrieval requires an encoder")
    require_vector_search_ready(
        session, model_name=encoder.model_name, dimensions=encoder.dimensions
    )
    query_embedding = encoder.encode([query], batch_size=1)[0]
    # HNSW returns at most ef_search rows, so leaving it unset caps the vector
    # candidate list at the server default of 40 however many candidate_limit asks
    # for — measured in Phase 1 and confirmed in Phase 4 (40 rows even at
    # candidate_limit=200). Defaulting ef_search to the pool size makes
    # candidate_limit mean what it says. Evidence:
    # docs/evidence/phase-8-fusion-heldout-v2.json.
    effective_ef = ef_search if ef_search is not None else max(candidate_limit, 40)
    vector_hits = vector_search(
        session, query_embedding, limit=candidate_limit, ef_search=effective_ef
    )
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
    keyword_hits = keyword_search(
        session, query, limit=candidate_limit, strategy=keyword_strategy
    )
    rankings = {"keyword": keyword_hits, "vector": vector_hits}
    fused = (
        convex_fusion(rankings, alpha=alpha)
        if fusion == "convex"
        else reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights)
    )
    if mode == "hybrid":
        return distinct_papers(fused, limit=limit)

    pooled = distinct_papers(fused, limit=pool)
    return _rerank(query, pooled, top_k=limit, reranker=reranker)

