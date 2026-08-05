"""Idempotent persistence for arXiv papers and their deterministic chunks."""

from __future__ import annotations

import re

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .arxiv import ArxivPaper
from .chunking import chunks_for_paper
from .models import Chunk, EmbeddingRun, Paper


def upsert_paper(session: Session, paper: ArxivPaper) -> tuple[int, int]:
    statement = (
        insert(Paper)
        .values(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
            authors=list(paper.authors),
            categories=list(paper.categories),
            primary_category=paper.primary_category,
            published_at=paper.published_at,
            updated_at=paper.updated_at,
            source_url=paper.source_url,
        )
        .on_conflict_do_update(
            index_elements=[Paper.arxiv_id],
            set_={
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": list(paper.authors),
                "categories": list(paper.categories),
                "primary_category": paper.primary_category,
                "published_at": paper.published_at,
                "updated_at": paper.updated_at,
                "source_url": paper.source_url,
            },
        )
        .returning(Paper.id)
    )
    paper_id = int(session.execute(statement).scalar_one())
    session.execute(delete(Chunk).where(Chunk.paper_id == paper_id))
    chunks = chunks_for_paper(paper)
    session.add_all(
        Chunk(paper_id=paper_id, ordinal=chunk.ordinal, text=chunk.text)
        for chunk in chunks
    )
    return paper_id, len(chunks)


def ingest_papers(session: Session, papers: list[ArxivPaper]) -> dict[str, int]:
    chunk_count = 0
    for paper in papers:
        _, saved_chunks = upsert_paper(session, paper)
        chunk_count += saved_chunks
    return {"papers": len(papers), "chunks": chunk_count}


def stored_arxiv_ids(session: Session) -> tuple[str, ...]:
    return tuple(session.execute(select(Paper.arxiv_id).order_by(Paper.arxiv_id)).scalars())


def unembedded_chunks(session: Session, *, limit: int) -> list[tuple[int, str]]:
    rows = session.execute(
        select(Chunk.id, Chunk.text)
        .where(Chunk.embedding.is_(None))
        .order_by(Chunk.id)
        .limit(limit)
    ).all()
    return [(int(row.id), row.text) for row in rows]


def save_embeddings(
    session: Session, chunk_ids: list[int], vectors: list[list[float]]
) -> None:
    if len(chunk_ids) != len(vectors):
        raise ValueError("chunk ID and embedding counts differ")
    for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
        session.execute(
            update(Chunk).where(Chunk.id == chunk_id).values(embedding=vector)
        )


def embedding_counts(session: Session) -> tuple[int, int]:
    total = int(session.scalar(select(func.count()).select_from(Chunk)) or 0)
    embedded = int(
        session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
        )
        or 0
    )
    return embedded, total


def record_embedding_run(
    session: Session,
    *,
    model_name: str,
    dimensions: int,
    corpus_arxiv_ids_sha256: str,
    embedded_chunk_count: int,
    total_chunk_count: int,
) -> tuple[int, bool]:
    latest = session.scalar(select(EmbeddingRun).order_by(EmbeddingRun.id.desc()).limit(1))
    expected = (model_name, dimensions, corpus_arxiv_ids_sha256, total_chunk_count)
    if latest is not None:
        actual = (
            latest.model_name,
            latest.dimensions,
            latest.corpus_arxiv_ids_sha256,
            latest.total_chunk_count,
        )
        if latest.status == "running":
            if actual != expected:
                raise ValueError("an incompatible embedding run is already in progress")
            return int(latest.id), True
        if (
            latest.status == "complete"
            and actual == expected
            and embedded_chunk_count == total_chunk_count
        ):
            return int(latest.id), False
    if embedded_chunk_count:
        raise ValueError(
            "database contains embeddings without a compatible resumable run; "
            "refusing to mix model provenance"
        )
    run = EmbeddingRun(
        model_name=model_name,
        dimensions=dimensions,
        normalized=True,
        corpus_arxiv_ids_sha256=corpus_arxiv_ids_sha256,
        status="running",
        embedded_chunk_count=0,
        total_chunk_count=total_chunk_count,
    )
    session.add(run)
    session.flush()
    return int(run.id), True


def update_embedding_run(
    session: Session, run_id: int, *, embedded_chunk_count: int, complete: bool
) -> None:
    session.execute(
        update(EmbeddingRun)
        .where(EmbeddingRun.id == run_id)
        .values(
            embedded_chunk_count=embedded_chunk_count,
            status="complete" if complete else "running",
        )
    )


def vector_search(
    session: Session, query_embedding: list[float], *, limit: int
) -> list[dict[str, object]]:
    distance = Chunk.embedding.cosine_distance(query_embedding)
    rows = session.execute(
        select(
            Paper.arxiv_id,
            Paper.title,
            Paper.source_url,
            Chunk.id.label("chunk_id"),
            Chunk.ordinal,
            Chunk.text,
            (1.0 - distance).label("score"),
        )
        .join(Paper, Paper.id == Chunk.paper_id)
        .where(Chunk.embedding.is_not(None))
        .order_by(distance, Chunk.id)
        .limit(limit)
    ).all()
    return [
        {
            "arxiv_id": row.arxiv_id,
            "title": row.title,
            "source_url": row.source_url,
            "chunk_id": int(row.chunk_id),
            "ordinal": int(row.ordinal),
            "text": row.text,
            "score": float(row.score),
        }
        for row in rows
    ]


def keyword_search(
    session: Session, query_text: str, *, limit: int
) -> list[dict[str, object]]:
    terms = tuple(
        dict.fromkeys(
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", query_text)
            if len(token) >= 3
        )
    )
    if not terms:
        return []
    query = func.websearch_to_tsquery("english", " OR ".join(terms))
    rank = func.ts_rank_cd(Chunk.search_vector, query)
    rows = session.execute(
        select(
            Paper.arxiv_id,
            Paper.title,
            Paper.source_url,
            Chunk.id.label("chunk_id"),
            Chunk.ordinal,
            Chunk.text,
            rank.label("score"),
        )
        .join(Paper, Paper.id == Chunk.paper_id)
        .where(Chunk.search_vector.op("@@")(query))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    ).all()
    return [
        {
            "arxiv_id": row.arxiv_id,
            "title": row.title,
            "source_url": row.source_url,
            "chunk_id": int(row.chunk_id),
            "ordinal": int(row.ordinal),
            "text": row.text,
            "score": float(row.score),
        }
        for row in rows
    ]


def require_vector_search_ready(
    session: Session, *, model_name: str, dimensions: int
) -> None:
    latest = session.scalar(select(EmbeddingRun).order_by(EmbeddingRun.id.desc()).limit(1))
    embedded, total = embedding_counts(session)
    if latest is None or latest.status != "complete" or embedded != total or total == 0:
        raise ValueError(
            f"vector search is not ready: embedded {embedded}/{total} chunks"
        )
    if latest.model_name != model_name or latest.dimensions != dimensions:
        raise ValueError(
            "query encoder does not match stored embedding provenance: "
            f"stored={latest.model_name}/{latest.dimensions}, "
            f"query={model_name}/{dimensions}"
        )
