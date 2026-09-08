"""Add a field-weighted search vector alongside the original (brief Part F, F2-iii).

Deliberately **additive**. The brief specified replacing the generated
`search_vector` column, but replacing it also changes `ts_rank_cd` for the
existing `"or"` strategy: today every lexeme is weight D, and a weighted column
makes title lexemes A and body B, so default rank weights `{0.1,0.2,0.4,1.0}`
would silently reorder the current default. That would (a) stop the frozen v3
baseline's keyword rows from reproducing against the live database and (b) make
"cascade" vs "cascade+weights" unmeasurable. Keeping both columns preserves A7 —
the default stays `"or"` until Phase 4 chooses, and the choice stays measurable.

Chunk text is `title + E'\n\n' + abstract_chunk`; verified before writing this
migration that `split_part(text, E'\n\n', 1)` equals the paper title for all
2,039 rows.
"""

from alembic import op


revision = "20260908_0003"
down_revision = "20260805_0002"
branch_labels = None
depends_on = None


WEIGHTED_EXPRESSION = (
    "setweight(to_tsvector('english', split_part(text, E'\\n\\n', 1)), 'A') || "
    "setweight(to_tsvector('english', "
    "substr(text, length(split_part(text, E'\\n\\n', 1)) + 3)), 'B')"
)


def upgrade() -> None:
    op.execute(
        f"""
        ALTER TABLE chunks
        ADD COLUMN search_vector_weighted tsvector
        GENERATED ALWAYS AS ({WEIGHTED_EXPRESSION}) STORED
        """
    )
    op.execute(
        "CREATE INDEX ix_chunks_search_vector_weighted_gin "
        "ON chunks USING gin (search_vector_weighted)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_search_vector_weighted_gin")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS search_vector_weighted")
