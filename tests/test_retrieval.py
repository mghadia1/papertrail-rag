import pytest

from papertrail.retrieval import distinct_papers, reciprocal_rank_fusion


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
