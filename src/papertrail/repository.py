"""Idempotent persistence for arXiv papers and their deterministic chunks."""

from __future__ import annotations

import re

from sqlalchemy import ARRAY, REAL, cast, delete, func, select, text, update
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
    session: Session,
    query_embedding: list[float],
    *,
    limit: int,
    exact: bool = False,
    ef_search: int | None = None,
) -> list[dict[str, object]]:
    """Cosine vector search over embedded chunks.

    By default this uses the pgvector HNSW index (m=16, ef_construction=64,
    vector_cosine_ops) at the server's ``hnsw.ef_search`` (default 40).

    ``exact=True`` forces an exact sequential scan by disabling index and bitmap
    scans for this transaction, giving the true nearest neighbours (the ground
    truth for a recall study). ``ef_search`` overrides the HNSW search breadth.

    Postgres facts this relies on (A12/A13 of the execution brief):
    - ``SET LOCAL`` applies only within the current transaction, so it is issued
      on this ``session`` right before the query and must not be committed away.
    - HNSW returns at most ``hnsw.ef_search`` rows, so ``ef_search`` must be
      ``>= limit`` or results are silently truncated; enforced below.
    - ``hnsw.ef_search`` is capped at 1000.
    ``exact`` and ``ef_search`` are mutually exclusive knobs; ``exact`` wins.
    """
    if exact:
        session.execute(text("SET LOCAL enable_indexscan = off"))
        session.execute(text("SET LOCAL enable_bitmapscan = off"))
    elif ef_search is not None:
        if not 1 <= ef_search <= 1000:
            raise ValueError("ef_search must be in [1, 1000]")
        if ef_search < limit:
            raise ValueError("ef_search must be >= limit or results are truncated")
        # SET does not accept bind parameters; ef_search is validated to an int in
        # [1, 1000] above, so this interpolation is safe.
        session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
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


# ts_rank_cd normalization flags (Postgres). 0 ignores document length; 1 divides
# by 1+log(length); 2 divides by length. Flag 32 (rank/(rank+1)) is a monotonic
# squash and cannot reorder results, so it is not a length normalization — see the
# Part F, F2-iii note in docs/lab-notes.md.
RANK_NORMALIZATIONS = (0, 1, 2, 16, 32)

# ts_rank_cd weights are ordered {D, C, B, A}; these are Postgres's defaults, made
# explicit so the weighted-column variant states what it applies.
_RANK_WEIGHTS = [0.1, 0.2, 0.4, 1.0]


def keyword_search(
    session: Session,
    query_text: str,
    *,
    limit: int,
    weighted: bool = False,
    normalization: int = 0,
) -> list[dict[str, object]]:
    """OR-of-terms full-text search.

    ``weighted`` ranks against the field-weighted ``search_vector_weighted``
    column (title lexemes A, body B) instead of the original unweighted column;
    ``normalization`` is the ``ts_rank_cd`` normalization flag. Both default to
    the original behaviour, so the frozen ``"or"`` results stay bit-identical
    (brief A7: the default does not change until Phase 4 chooses).
    """
    if normalization not in RANK_NORMALIZATIONS:
        raise ValueError(f"unsupported ts_rank_cd normalization flag: {normalization}")
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
    if weighted:
        # Postgres needs the weight array typed as real[].
        rank = func.ts_rank_cd(
            cast(_RANK_WEIGHTS, ARRAY(REAL)),
            Chunk.search_vector_weighted,
            query,
            normalization,
        )
    elif normalization:
        rank = func.ts_rank_cd(Chunk.search_vector, query, normalization)
    else:
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
        .where(
            (Chunk.search_vector_weighted if weighted else Chunk.search_vector).op("@@")(
                query
            )
        )
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
