"""Idempotent persistence for arXiv papers and their deterministic chunks."""

from __future__ import annotations

import re

from sqlalchemy import ARRAY, REAL, cast, delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .arxiv import ArxivPaper
from .chunking import chunks_for_paper
from .models import EMBEDDING_COLUMNS, Chunk, EmbeddingRun, Paper


def embedding_column(name: str):
    """Resolve a per-model embedding column against the whitelist (brief H trap).

    Every caller must pass the column explicitly; nothing interpolates a caller's
    string into SQL, so an unknown column is a clear error rather than an injection
    surface or a silent read of the wrong vectors.
    """
    if name not in EMBEDDING_COLUMNS:
        raise ValueError(
            f"unknown embedding column {name!r}; expected one of {sorted(EMBEDDING_COLUMNS)}"
        )
    return getattr(Chunk, name)


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


def unembedded_chunks(
    session: Session, *, limit: int, column: str = "embedding"
) -> list[tuple[int, str]]:
    target = embedding_column(column)
    rows = session.execute(
        select(Chunk.id, Chunk.text)
        .where(target.is_(None))
        .order_by(Chunk.id)
        .limit(limit)
    ).all()
    return [(int(row.id), row.text) for row in rows]


def save_embeddings(
    session: Session,
    chunk_ids: list[int],
    vectors: list[list[float]],
    *,
    column: str = "embedding",
) -> None:
    if len(chunk_ids) != len(vectors):
        raise ValueError("chunk ID and embedding counts differ")
    embedding_column(column)  # validate before building any statement
    for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
        session.execute(
            update(Chunk).where(Chunk.id == chunk_id).values(**{column: vector})
        )


def embedding_counts(session: Session, *, column: str = "embedding") -> tuple[int, int]:
    target = embedding_column(column)
    total = int(session.scalar(select(func.count()).select_from(Chunk)) or 0)
    embedded = int(
        session.scalar(
            select(func.count()).select_from(Chunk).where(target.is_not(None))
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
    column_name: str = "embedding",
    query_prefix: str = "",
    passage_prefix: str = "",
) -> tuple[int, bool]:
    """Find or open the run for this (model, column), returning (id, should_run).

    Keyed on the model/column pair rather than "the latest run", because Phase 5
    keeps several models side by side in their own columns. The prefixes are part of
    a run's provenance: the same model indexed with a different passage prefix is a
    different run and must not silently resume the old one.
    """
    embedding_column(column_name)
    existing = session.scalar(
        select(EmbeddingRun).where(
            EmbeddingRun.model_name == model_name,
            EmbeddingRun.column_name == column_name,
        )
    )
    expected = (
        dimensions,
        corpus_arxiv_ids_sha256,
        total_chunk_count,
        query_prefix,
        passage_prefix,
    )
    if existing is not None:
        actual = (
            existing.dimensions,
            existing.corpus_arxiv_ids_sha256,
            existing.total_chunk_count,
            existing.query_prefix,
            existing.passage_prefix,
        )
        if actual != expected:
            raise ValueError(
                f"an incompatible embedding run already exists for {model_name} in "
                f"{column_name}: stored={actual}, requested={expected}"
            )
        if existing.status == "complete" and embedded_chunk_count == total_chunk_count:
            return int(existing.id), False
        return int(existing.id), True
    if embedded_chunk_count:
        raise ValueError(
            f"{column_name} already contains embeddings without a matching run; "
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
        column_name=column_name,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
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
    column: str = "embedding",
) -> list[dict[str, object]]:
    """Cosine vector search over embedded chunks.

    ``column`` selects which per-model embedding column to search; the study must
    pass it explicitly so a run can never read another model's vectors (brief H
    trap). Each column has its own HNSW index.

    By default this uses the pgvector HNSW index (m=16, ef_construction=64,
    vector_cosine_ops) at the server's ``hnsw.ef_search`` (default 40).

    ``exact=True`` forces an exact sequential scan by disabling index and bitmap
    scans for this transaction, giving the true nearest neighbours (the ground
    truth for a recall study). ``ef_search`` overrides the HNSW search breadth.

    Postgres facts this relies on (A12/A12b/A13 of the execution brief):
    - ``SET LOCAL`` applies for the rest of the current transaction, not just the
      next statement. So every GUC this helper touches is set on **every** call —
      either to a value or back to ``DEFAULT`` — and a call can never inherit a
      previous call's setting. Omitting the reset caused a real measurement error:
      a Phase 4 run set ``ef_search=200`` for one configuration and the next
      configuration in the same session silently inherited it, which made a
      truncating baseline look untruncated (A12b).
    - HNSW returns at most ``hnsw.ef_search`` rows, so ``ef_search`` must be
      ``>= limit`` or results are silently truncated; enforced below.
    - ``hnsw.ef_search`` is capped at 1000.
    ``exact`` and ``ef_search`` are mutually exclusive knobs; ``exact`` wins.
    """
    # Validate before touching the session, so bad input never leaves half-applied
    # settings behind in the caller's transaction.
    use_ef = not exact and ef_search is not None
    if use_ef:
        if not 1 <= ef_search <= 1000:
            raise ValueError("ef_search must be in [1, 1000]")
        if ef_search < limit:
            raise ValueError("ef_search must be >= limit or results are truncated")

    if exact:
        session.execute(text("SET LOCAL enable_indexscan = off"))
        session.execute(text("SET LOCAL enable_bitmapscan = off"))
    else:
        session.execute(text("SET LOCAL enable_indexscan TO DEFAULT"))
        session.execute(text("SET LOCAL enable_bitmapscan TO DEFAULT"))
    if use_ef:
        # SET does not accept bind parameters; ef_search is validated to an int in
        # [1, 1000] above, so this interpolation is safe.
        session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    else:
        session.execute(text("SET LOCAL hnsw.ef_search TO DEFAULT"))
    target = embedding_column(column)
    distance = target.cosine_distance(query_embedding)
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
        .where(target.is_not(None))
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


KEYWORD_STRATEGIES = ("or", "cascade", "cascade_phrase")

_QUOTED_SPAN_RE = re.compile(r"[\"“]([^\"”]{3,})[\"”]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def phrase_candidates(query_text: str) -> list[str]:
    """Phrases worth an exact-order match: quoted spans, then adjacent capitalised
    word pairs (a cheap proxy for a named entity such as "Chain of Thought")."""
    phrases: list[str] = [
        span.strip() for span in _QUOTED_SPAN_RE.findall(query_text) if span.strip()
    ]
    words = _WORD_RE.findall(query_text)
    for first, second in zip(words, words[1:]):
        if first[:1].isupper() and second[:1].isupper():
            phrases.append(f"{first} {second}")
    return list(dict.fromkeys(phrases))


def keyword_terms(query_text: str) -> tuple[str, ...]:
    """Query terms: the shared regex, lower-cased, <3 chars dropped, de-duplicated."""
    return tuple(
        dict.fromkeys(
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", query_text)
            if len(token) >= 3
        )
    )


def _keyword_rows(
    session: Session,
    query,
    *,
    limit: int,
    weighted: bool,
    normalization: int,
) -> list[dict[str, object]]:
    """Run one tsquery and return ranked chunk rows."""
    column = Chunk.search_vector_weighted if weighted else Chunk.search_vector
    if weighted:
        # Postgres needs the weight array typed as real[].
        rank = func.ts_rank_cd(
            cast(_RANK_WEIGHTS, ARRAY(REAL)), column, query, normalization
        )
    elif normalization:
        rank = func.ts_rank_cd(column, query, normalization)
    else:
        rank = func.ts_rank_cd(column, query)
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
        .where(column.op("@@")(query))
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


def keyword_search(
    session: Session,
    query_text: str,
    *,
    limit: int,
    weighted: bool = False,
    normalization: int = 0,
    strategy: str = "or",
) -> list[dict[str, object]]:
    """Full-text search over chunk text.

    ``strategy="or"`` (the default, unchanged) is one OR-of-all-terms query.
    ``strategy="cascade"`` runs an AND of every term first and, if that returns
    fewer than ``limit`` rows, fills the remainder from the OR query, keeping the
    AND rows ahead of the OR rows.

    ``weighted`` ranks against the field-weighted ``search_vector_weighted``
    column (title lexemes A, body B) instead of the original unweighted column;
    ``normalization`` is the ``ts_rank_cd`` normalization flag. Every argument
    defaults to the original behaviour, so the frozen ``"or"`` results stay
    bit-identical (brief A7: the default does not change until Phase 4 chooses).
    """
    if normalization not in RANK_NORMALIZATIONS:
        raise ValueError(f"unsupported ts_rank_cd normalization flag: {normalization}")
    if strategy not in KEYWORD_STRATEGIES:
        raise ValueError(f"unsupported keyword strategy: {strategy!r}")
    terms = keyword_terms(query_text)
    if not terms:
        return []

    or_query = func.websearch_to_tsquery("english", " OR ".join(terms))
    if strategy == "or":
        return _keyword_rows(
            session, or_query, limit=limit, weighted=weighted, normalization=normalization
        )

    ordered: list[dict[str, object]] = []
    seen: set[int] = set()

    def extend(rows: list[dict[str, object]]) -> None:
        for row in rows:
            chunk_id = int(row["chunk_id"])
            if chunk_id not in seen:
                seen.add(chunk_id)
                ordered.append(row)

    if strategy == "cascade_phrase":
        # Exact-order phrase matches rank ahead of everything else. phraseto_tsquery
        # yields an empty query for an all-stop-word phrase, which simply matches
        # nothing.
        for phrase in phrase_candidates(query_text):
            extend(
                _keyword_rows(
                    session,
                    func.phraseto_tsquery("english", phrase),
                    limit=limit,
                    weighted=weighted,
                    normalization=normalization,
                )
            )
            if len(ordered) >= limit:
                return ordered[:limit]

    # Cascade: AND of every term first. Each term is quoted so a hyphenated token
    # ("state-of-the-art") is parsed as a lexeme rather than as an operator. An
    # all-stop-word query yields an empty tsquery, which matches nothing and simply
    # falls through to the OR fill below rather than erroring.
    and_query = func.to_tsquery("english", " & ".join(f"'{term}'" for term in terms))
    extend(
        _keyword_rows(
            session, and_query, limit=limit, weighted=weighted, normalization=normalization
        )
    )
    if len(ordered) >= limit:
        return ordered[:limit]

    extend(
        _keyword_rows(
            session, or_query, limit=limit, weighted=weighted, normalization=normalization
        )
    )
    return ordered[:limit]


def require_vector_search_ready(
    session: Session, *, model_name: str, dimensions: int, column: str = "embedding"
) -> None:
    """Assert the run for this (model, column) is complete before querying it.

    Checks the ``embedding_runs`` row for the chosen model and column rather than
    "the latest run" (brief H1), so evaluating one model cannot pass because a
    different model finished, and an unfinished column fails loudly.
    """
    embedding_column(column)
    run = session.scalar(
        select(EmbeddingRun).where(
            EmbeddingRun.model_name == model_name,
            EmbeddingRun.column_name == column,
        )
    )
    embedded, total = embedding_counts(session, column=column)
    if run is None:
        raise ValueError(
            f"no embedding run recorded for {model_name} in {column}; "
            f"embedded {embedded}/{total} chunks"
        )
    if run.status != "complete" or embedded != total or total == 0:
        raise ValueError(
            f"vector search is not ready for {model_name} in {column}: "
            f"run status {run.status}, embedded {embedded}/{total} chunks"
        )
    if run.dimensions != dimensions:
        raise ValueError(
            "query encoder does not match stored embedding provenance: "
            f"stored={run.model_name}/{run.dimensions} in {column}, "
            f"query={model_name}/{dimensions}"
        )
