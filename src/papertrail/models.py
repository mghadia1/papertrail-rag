"""SQLAlchemy schema for papers, chunks, vector search, and full-text search."""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


EMBEDDING_DIMENSIONS = 384

# Per-model embedding columns for the Phase 5 encoder ablation (brief Part H).
# "embedding" is MiniLM's frozen column and is never re-embedded. The mapping is a
# whitelist: every column parameter is validated against it, so a column name can
# never reach SQL unchecked.
EMBEDDING_COLUMNS: dict[str, int] = {
    "embedding": 384,
    "embedding_bge_small": 384,
    "embedding_e5_small": 384,
    "embedding_e5_small_noprefix": 384,
    "embedding_gte_small": 384,
    "embedding_bge_base": 768,
}


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    categories: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    primary_category: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("paper_id", "ordinal", name="uq_chunks_paper_ordinal"),
        Index(
            "ix_chunks_embedding_hnsw_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_search_vector_gin", "search_vector", postgresql_using="gin"),
        *(
            Index(
                f"ix_chunks_{column}_hnsw_cosine",
                column,
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={column: "vector_cosine_ops"},
            )
            for column in EMBEDDING_COLUMNS
            if column != "embedding"
        ),
        Index(
            "ix_chunks_search_vector_weighted_gin",
            "search_vector_weighted",
            postgresql_using="gin",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        VECTOR(EMBEDDING_DIMENSIONS), nullable=True
    )
    # Phase 5 ablation columns; all nullable so an un-run model is simply empty.
    embedding_bge_small: Mapped[list[float] | None] = mapped_column(VECTOR(384), nullable=True)
    embedding_e5_small: Mapped[list[float] | None] = mapped_column(VECTOR(384), nullable=True)
    embedding_e5_small_noprefix: Mapped[list[float] | None] = mapped_column(VECTOR(384), nullable=True)
    embedding_gte_small: Mapped[list[float] | None] = mapped_column(VECTOR(384), nullable=True)
    embedding_bge_base: Mapped[list[float] | None] = mapped_column(VECTOR(768), nullable=True)
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(text, ''))", persisted=True),
        nullable=False,
    )
    # Field-weighted variant kept alongside the original so the current "or"
    # strategy stays bit-identical and both remain measurable (Part F, F2-iii).
    # Chunk text is title + E'\n\n' + abstract_chunk.
    search_vector_weighted: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', split_part(text, E'\\n\\n', 1)), 'A') || "
            "setweight(to_tsvector('english', "
            "substr(text, length(split_part(text, E'\\n\\n', 1)) + 3)), 'B')",
            persisted=True,
        ),
        nullable=False,
    )

    paper: Mapped[Paper] = relationship(back_populates="chunks")


class EmbeddingRun(Base):
    __tablename__ = "embedding_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    normalized: Mapped[bool] = mapped_column(nullable=False)
    corpus_arxiv_ids_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    embedded_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Where this run's vectors live, and the prefixes it was built with.
    column_name: Mapped[str] = mapped_column(Text, nullable=False, server_default="embedding")
    query_prefix: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    passage_prefix: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
