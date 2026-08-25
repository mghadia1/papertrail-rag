"""Tests for two-stage retrieval and cross-encoder reranking."""

from __future__ import annotations

import pytest

from papertrail.reranking import LexicalSemanticReranker, CrossEncoderReranker


def test_lexical_semantic_reranker_scores_and_orders_by_relevance():
    reranker = LexicalSemanticReranker()

    candidates = [
        {
            "chunk_id": 1,
            "arxiv_id": "2301.0001v1",
            "title": "Unrelated Physics Paper on Thermodynamics",
            "text": "Heat transfer and entropy calculations.",
            "score": 0.03,
        },
        {
            "chunk_id": 2,
            "arxiv_id": "2301.0002v1",
            "title": "Scaled Dot-Product Attention in Transformer Networks",
            "text": "We present multi-head attention and scaled dot-product mechanisms for transformer architectures.",
            "score": 0.02,
        },
        {
            "chunk_id": 3,
            "arxiv_id": "2301.0003v1",
            "title": "General Survey of Deep Learning",
            "text": "An overview of neural networks and architectures including attention.",
            "score": 0.025,
        },
    ]

    query = "scaled dot-product attention transformer architectures"
    reranked = reranker.rerank(query, candidates, top_k=2)

    assert len(reranked) == 2
    # The second paper is an exact title and text match for the query, so it should be reranked to rank 1
    assert reranked[0]["arxiv_id"] == "2301.0002v1"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]


def test_reranker_rejects_invalid_top_k():
    reranker = LexicalSemanticReranker()
    with pytest.raises(ValueError, match="top_k must be positive"):
        reranker.rerank("query", [{"chunk_id": 1}], top_k=0)


def test_cross_encoder_fallback_works_when_model_absent():
    reranker = CrossEncoderReranker(model_name="nonexistent-model-identifier")
    candidates = [
        {
            "chunk_id": 1,
            "arxiv_id": "2401.0001v1",
            "title": "Graph Neural Networks for Molecule Generation",
            "text": "Using GNNs to design new molecules.",
            "score": 0.01,
        }
    ]
    reranked = reranker.rerank("molecule generation graph neural networks", candidates, top_k=1)
    assert len(reranked) == 1
    assert reranked[0]["arxiv_id"] == "2401.0001v1"
