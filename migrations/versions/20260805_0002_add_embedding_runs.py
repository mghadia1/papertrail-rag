"""Record the exact model and corpus used for each completed embedding run."""

from alembic import op


revision = "20260805_0002"
down_revision = "20260805_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE embedding_runs (
            id BIGSERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            normalized BOOLEAN NOT NULL,
            corpus_arxiv_ids_sha256 TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('running', 'complete')),
            embedded_chunk_count INTEGER NOT NULL,
            total_chunk_count INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_runs")
