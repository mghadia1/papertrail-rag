import pytest

from papertrail import retrieval
from papertrail.retrieval import distinct_papers, reciprocal_rank_fusion, retrieve


def hit(chunk_id: int, arxiv_id: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id,
        "title": arxiv_id,
        "source_url": f"https://arxiv.org/abs/{arxiv_id}",
        "ordinal": 0,
        "text": arxiv_id,
        "score": 1.0,
    }


def test_rrf_math_rewards_results_present_in_both_rankings() -> None:
    fused = reciprocal_rank_fusion(
        {
            "vector": [hit(1, "a1v1"), hit(2, "a2v1")],
            "keyword": [hit(2, "a2v1"), hit(3, "a3v1")],
        },
        k=60,
    )
    assert fused[0]["chunk_id"] == 2
    assert fused[0]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0]["component_ranks"] == {"keyword": 1, "vector": 2}


def test_rrf_ties_are_deterministic() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": [hit(2, "b1v1")], "keyword": [hit(1, "a1v1")]}, k=60
    )
    assert [item["arxiv_id"] for item in fused] == ["a1v1", "b1v1"]


def test_rrf_rejects_duplicate_chunks_within_one_ranking() -> None:
    with pytest.raises(ValueError, match="duplicate chunk"):
        reciprocal_rank_fusion(
            {"vector": [hit(1, "a1v1"), hit(1, "a1v1")]}, k=60
        )


def test_distinct_papers_keeps_best_chunk_per_paper() -> None:
    selected = distinct_papers(
        [hit(1, "a1v1"), hit(2, "a1v1"), hit(3, "a2v1")], limit=2
    )
    assert [item["chunk_id"] for item in selected] == [1, 3]


class _FakeEncoder:
    model_name = "fake"
    dimensions = 3

    def encode(self, texts, batch_size=1):
        return [[0.0, 0.0, 0.0] for _ in texts]


class _IdentityReranker:
    """Reranker that preserves order; lets us assert which stage fed it."""

    model_name = "identity-reranker"
    max_length = None

    def rerank(self, query, candidates, *, top_k):
        return [dict(c) for c in candidates[:top_k]]


def _patch_searches(monkeypatch, *, keyword_raises: bool):
    monkeypatch.setattr(retrieval, "require_vector_search_ready", lambda *a, **k: None)
    vector_hits = [hit(i, f"v{i}") for i in range(1, 61)]
    monkeypatch.setattr(retrieval, "vector_search", lambda *a, **k: vector_hits)

    def _keyword(*a, **k):
        if keyword_raises:
            raise AssertionError("keyword_search must not run on the vector_rerank path")
        return [hit(i, f"k{i}") for i in range(1, 61)]

    monkeypatch.setattr(retrieval, "keyword_search", _keyword)
    return vector_hits


def test_vector_rerank_never_calls_keyword_search(monkeypatch) -> None:
    _patch_searches(monkeypatch, keyword_raises=True)
    results = retrieve(
        session=object(),
        query="q",
        mode="vector_rerank",
        limit=5,
        encoder=_FakeEncoder(),
        reranker=_IdentityReranker(),
        rerank_pool=20,
    )
    assert [r["arxiv_id"] for r in results] == [f"v{i}" for i in range(1, 6)]
    # The whole point of the mode: it must not touch keyword_search (E1).
    assert all(r["rerank_pool_size"] == 20 for r in results)


def test_rerank_pool_controls_first_stage_depth(monkeypatch) -> None:
    _patch_searches(monkeypatch, keyword_raises=True)
    results = retrieve(
        session=object(),
        query="q",
        mode="vector_rerank",
        limit=5,
        encoder=_FakeEncoder(),
        reranker=_IdentityReranker(),
        rerank_pool=50,
    )
    assert all(r["rerank_pool_size"] == 50 for r in results)


def test_rerank_pool_below_limit_is_rejected(monkeypatch) -> None:
    _patch_searches(monkeypatch, keyword_raises=False)
    with pytest.raises(ValueError, match="rerank_pool must be at least limit"):
        retrieve(
            session=object(),
            query="q",
            mode="hybrid_rerank",
            limit=10,
            encoder=_FakeEncoder(),
            reranker=_IdentityReranker(),
            rerank_pool=5,
        )
