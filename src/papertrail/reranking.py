"""Two-stage retrieval and Cross-Encoder reranking for PaperTrail."""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    model_name: str

    def rerank(
        self, query: str, candidates: list[dict[str, object]], *, top_k: int
    ) -> list[dict[str, object]]:
        """Rerank candidates using cross-encoder scoring."""


class LexicalSemanticReranker:
    """A deterministic, zero-external-dependency cross-scoring reranker.
    
    Scores candidates by combining exact query term density, title matching boost,
    and reciprocal rank scores to simulate high-precision cross-attention scoring
    in environments without torch/transformers installed.
    """

    def __init__(self, model_name: str = "lexical-semantic-cross-v1") -> None:
        self.model_name = model_name

    def score_pair(self, query: str, title: str, text: str) -> float:
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        if not query_terms:
            return 0.0

        title_lower = title.lower()
        text_lower = text.lower()

        # Title match score (high weight)
        title_hits = sum(1 for term in query_terms if term in title_lower)
        title_score = title_hits / len(query_terms)

        # Body match density (phrase and term matches)
        body_hits = sum(text_lower.count(term) for term in query_terms)
        body_score = min(1.0, body_hits / (len(query_terms) * 3))

        # Combined cross-score
        return 0.6 * title_score + 0.4 * body_score

    def rerank(
        self, query: str, candidates: list[dict[str, object]], *, top_k: int
    ) -> list[dict[str, object]]:
        if not candidates:
            return []
        if top_k < 1:
            raise ValueError("top_k must be positive")

        scored_candidates: list[dict[str, object]] = []
        for cand in candidates:
            title = str(cand.get("title", ""))
            text = str(cand.get("text", ""))
            base_score = float(cand.get("score", 0.0))
            cross_score = self.score_pair(query, title, text)
            
            # Combine initial retrieval prior with fine-grained cross-score
            final_score = 0.3 * base_score + 0.7 * cross_score
            item = dict(cand)
            item["rerank_score"] = cross_score
            item["score"] = final_score
            scored_candidates.append(item)

        scored_candidates.sort(
            key=lambda c: (
                -float(c["score"]),
                str(c.get("arxiv_id", "")),
                int(c.get("chunk_id", 0)),
            )
        )
        return scored_candidates[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker leveraging sentence-transformers when available."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._fallback = LexicalSemanticReranker(model_name=f"{model_name}-fallback")

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = False

    def rerank(
        self, query: str, candidates: list[dict[str, object]], *, top_k: int
    ) -> list[dict[str, object]]:
        if not candidates:
            return []
        if top_k < 1:
            raise ValueError("top_k must be positive")

        self._load_model()
        if not self._model:
            return self._fallback.rerank(query, candidates, top_k=top_k)

        pairs = [[query, f"{cand.get('title', '')} {cand.get('text', '')}"] for cand in candidates]
        scores = self._model.predict(pairs)

        scored_candidates: list[dict[str, object]] = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            item["score"] = float(score)
            scored_candidates.append(item)

        scored_candidates.sort(
            key=lambda c: (
                -float(c["score"]),
                str(c.get("arxiv_id", "")),
                int(c.get("chunk_id", 0)),
            )
        )
        return scored_candidates[:top_k]
