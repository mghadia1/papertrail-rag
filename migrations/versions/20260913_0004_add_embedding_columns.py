"""Per-model embedding columns for the Phase 5 encoder ablation (brief Part H).

Additive, like 20260908_0003. The frozen 384-d ``embedding`` column (MiniLM) is
never touched, so the v3 baseline keeps reproducing byte-for-byte and H4's check
is structural rather than a question of re-embedding determinism.

``embedding_runs`` gains ``column_name`` so a run records *where* its vectors live,
plus the two prefixes it was built with — forgetting a prefix at query time while
using it at index time is the trap this phase is designed to expose, so the run row
has to carry them.
"""

from alembic import op


revision = "20260913_0004"
down_revision = "20260908_0003"
branch_labels = None
depends_on = None

# column name -> dimensions
COLUMNS = {
    "embedding_bge_small": 384,
    "embedding_e5_small": 384,
    "embedding_e5_small_noprefix": 384,
    "embedding_gte_small": 384,
    "embedding_bge_base": 768,
}


def upgrade() -> None:
    for column, dimensions in COLUMNS.items():
        op.execute(f"ALTER TABLE chunks ADD COLUMN {column} vector({dimensions})")
        op.execute(
            f"CREATE INDEX ix_chunks_{column}_hnsw_cosine ON chunks "
            f"USING hnsw ({column} vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
    op.execute(
        "ALTER TABLE embedding_runs "
        "ADD COLUMN column_name TEXT NOT NULL DEFAULT 'embedding', "
        "ADD COLUMN query_prefix TEXT NOT NULL DEFAULT '', "
        "ADD COLUMN passage_prefix TEXT NOT NULL DEFAULT ''"
    )
    # A model/column pair is one run; this makes a duplicate run impossible rather
    # than merely unlikely.
    op.execute(
        "CREATE UNIQUE INDEX uq_embedding_runs_model_column "
        "ON embedding_runs (model_name, column_name)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_embedding_runs_model_column")
    op.execute(
        "ALTER TABLE embedding_runs "
        "DROP COLUMN IF EXISTS passage_prefix, "
        "DROP COLUMN IF EXISTS query_prefix, "
        "DROP COLUMN IF EXISTS column_name"
    )
    for column in COLUMNS:
        op.execute(f"DROP INDEX IF EXISTS ix_chunks_{column}_hnsw_cosine")
        op.execute(f"ALTER TABLE chunks DROP COLUMN IF EXISTS {column}")
