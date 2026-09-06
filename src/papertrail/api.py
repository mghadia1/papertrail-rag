"""FastAPI health and traceable vector-search surface."""

from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from . import __version__
from .database import session_scope
from .config import get_settings
from .embedding import get_encoder
from .generation import AnswerResult, answer_question, get_generator
from .retrieval import SearchMode, retrieve


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    resume_eligible: bool
    milestone: str


class SearchHit(BaseModel):
    arxiv_id: str
    title: str
    source_url: str
    chunk_id: int
    ordinal: int
    text: str
    score: float
    component_ranks: dict[str, int] | None = None


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    model: str | None
    results: list[SearchHit]


class NearestPaper(BaseModel):
    arxiv_id: str
    title: str
    source_url: str


class AnswerResponse(BaseModel):
    question: str
    answer: str | None
    abstained: bool
    abstain_reason: str | None
    retrieval_mode: str
    top_score: float | None
    threshold: float
    generator_model: str | None
    citations: tuple[str, ...]
    retrieved_arxiv_ids: tuple[str, ...]
    gate_signal_name: str = "rrf_top"
    gate_score: float | None = None
    # On abstention, the closest evidence to show instead of a bare refusal.
    nearest_papers: list[NearestPaper] = []


app = FastAPI(title="PaperTrail", version=__version__)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="papertrail",
        version=__version__,
        resume_eligible=False,
        milestone="evaluated-rag",
    )


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=2, max_length=500),
    mode: SearchMode = "hybrid",
    limit: int = Query(default=5, ge=1, le=50),
) -> SearchResponse:
    encoder = get_encoder() if mode != "keyword" else None
    try:
        with session_scope() as session:
            hits = retrieve(
                session, q, mode=mode, limit=limit, encoder=encoder
            )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return SearchResponse(
        query=q,
        mode=mode,
        model=encoder.model_name if encoder is not None else None,
        results=[SearchHit.model_validate(hit) for hit in hits],
    )


@app.get("/ask", response_model=AnswerResponse)
def ask(
    q: str = Query(min_length=2, max_length=500),
    top_k: int = Query(default=5, ge=1, le=10),
) -> AnswerResponse:
    settings = get_settings()
    try:
        with session_scope() as session:
            result: AnswerResult = answer_question(
                session,
                q,
                encoder=get_encoder(),
                generator=get_generator(),
                threshold=settings.abstain_threshold,
                top_k=top_k,
                gate_signal_name=settings.abstain_signal,
            )
    except (RuntimeError, ValueError, httpx.HTTPError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return AnswerResponse.model_validate(result.__dict__)
