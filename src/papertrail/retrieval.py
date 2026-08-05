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


SearchMode = Literal["vector", "keyword", "hybrid"]


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


def retrieve(
    session: Session,
    query: str,
    *,
    mode: SearchMode,
    limit: int,
    encoder: Encoder | None = None,
    rrf_k: int = 60,
) -> list[dict[str, object]]:
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
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
    if mode != "hybrid":
        raise ValueError(f"unsupported search mode: {mode}")
    keyword_hits = keyword_search(session, query, limit=candidate_limit)
    return distinct_papers(
        reciprocal_rank_fusion(
            {"keyword": keyword_hits, "vector": vector_hits}, k=rrf_k
        ),
        limit=limit,
    )
