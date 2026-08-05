"""Create papers, chunks, pgvector HNSW, and full-text GIN indexes."""

from alembic import op


revision = "20260805_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE papers (
            id BIGSERIAL PRIMARY KEY,
            arxiv_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            authors TEXT[] NOT NULL,
            categories TEXT[] NOT NULL,
            primary_category TEXT NOT NULL,
            published_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            source_url TEXT NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE chunks (
            id BIGSERIAL PRIMARY KEY,
            paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding vector(384),
            search_vector tsvector GENERATED ALWAYS AS
                (to_tsvector('english', coalesce(text, ''))) STORED,
            CONSTRAINT uq_chunks_paper_ordinal UNIQUE (paper_id, ordinal)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw_cosine ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.execute(
        "CREATE INDEX ix_chunks_search_vector_gin ON chunks USING gin (search_vector)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS papers")
    op.execute("DROP EXTENSION IF EXISTS vector")

