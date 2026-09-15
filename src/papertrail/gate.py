"""Abstention-gate confidence signals, shared by the study and production.

`signal_from_hits` computes one named signal from already-retrieved results, so
the gate study (`eval/tools/gate_signals.py`) and `answer_question` compute the
exact same number. Each signal declares its valid range: rrf/cos/sigmoid live in
[0, 1], keyword is non-negative, and the raw cross-encoder logits (`ce_top`,
`ce_margin`) are unbounded — so the threshold range check must be per-signal, not
a hardcoded [0, 1] (see D5 of the execution brief).
"""

from __future__ import annotations

import math
import statistics

from .embedding import Encoder
from .repository import keyword_search, require_vector_search_ready, vector_search
from .retrieval import retrieve

# (low, high); None means unbounded on that side.
SIGNAL_RANGES: dict[str, tuple[float | None, float | None]] = {
    "rrf_top": (0.0, 1.0),
    "cos_top": (0.0, 1.0),
    "cos_margin": (0.0, 1.0),
    "cos_mean_top3": (0.0, 1.0),
    "kw_top": (0.0, None),
    "ce_top": (None, None),
    "ce_margin": (0.0, None),
    "ce_sigmoid_top": (0.0, 1.0),
}


def signal_from_hits(
    name: str,
    *,
    hybrid_hits: list[dict] | None = None,
    vector_hits: list[dict] | None = None,
    keyword_hits: list[dict] | None = None,
    rerank_hits: list[dict] | None = None,
) -> float:
    if name == "rrf_top":
        return float(hybrid_hits[0]["score"]) if hybrid_hits else 0.0
    if name in ("cos_top", "cos_margin", "cos_mean_top3"):
        cos = [float(h["score"]) for h in (vector_hits or [])]
        if name == "cos_top":
            return cos[0] if cos else 0.0
        if name == "cos_margin":
            return (cos[0] - cos[1]) if len(cos) >= 2 else 0.0
        return statistics.fmean(cos[:3]) if cos else 0.0
    if name == "kw_top":
        return float(keyword_hits[0]["score"]) if keyword_hits else 0.0
    if name in ("ce_top", "ce_margin", "ce_sigmoid_top"):
        ce = [float(h["rerank_score"]) for h in (rerank_hits or [])]
        top = ce[0] if ce else 0.0
        if name == "ce_top":
            return top
        if name == "ce_margin":
            return (ce[0] - ce[1]) if len(ce) >= 2 else 0.0
        return 1.0 / (1.0 + math.exp(-top))
    raise ValueError(f"unknown gate signal: {name!r}")


def _retrievals_needed(name: str) -> set[str]:
    if name == "rrf_top":
        return {"hybrid"}
    if name.startswith("cos_"):
        return {"vector"}
    if name == "kw_top":
        return {"keyword"}
    if name.startswith("ce_"):
        return {"rerank"}
    raise ValueError(f"unknown gate signal: {name!r}")


def validate_threshold(name: str, threshold: float) -> None:
    if name not in SIGNAL_RANGES:
        raise ValueError(f"unknown gate signal: {name!r}")
    low, high = SIGNAL_RANGES[name]
    if low is not None and threshold < low:
        raise ValueError(f"threshold {threshold} is below the valid range for {name}")
    if high is not None and threshold > high:
        raise ValueError(f"threshold {threshold} is above the valid range for {name}")


def gate_signal(
    session,
    query: str,
    encoder: Encoder,
    name: str,
    *,
    hybrid_hits: list[dict] | None = None,
    embedding_column: str = "embedding",
) -> float:
    """Fetch only the retrievals the signal needs, then compute it.

    Pass ``hybrid_hits`` when the caller already ran hybrid retrieval (the common
    ``rrf_top`` case) to avoid a redundant query. ``embedding_column`` must be the
    column ``encoder`` indexed: every vector read goes through it, so a query
    embedding is never compared against another model's vectors (brief Part H).
    """
    needs = _retrievals_needed(name)
    hyb = hybrid_hits
    if "hybrid" in needs and hyb is None:
        hyb = retrieve(session, query, mode="hybrid", limit=5, encoder=encoder,
                       embedding_column=embedding_column)
    vec = None
    if "vector" in needs:
        require_vector_search_ready(
            session, model_name=encoder.model_name, dimensions=encoder.dimensions,
            column=embedding_column,
        )
        vec = vector_search(session, encoder.encode([query], batch_size=1)[0], limit=50,
                            column=embedding_column)
    kw = keyword_search(session, query, limit=50) if "keyword" in needs else None
    rr = (
        retrieve(session, query, mode="hybrid_rerank", limit=5, encoder=encoder,
                 embedding_column=embedding_column)
        if "rerank" in needs
        else None
    )
    return signal_from_hits(name, hybrid_hits=hyb, vector_hits=vec, keyword_hits=kw, rerank_hits=rr)
