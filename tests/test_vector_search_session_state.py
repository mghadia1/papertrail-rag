"""Integration tests that `vector_search` never leaks Postgres session state.

`SET LOCAL` lasts for the rest of the transaction, not one statement. A Phase 4
run set `ef_search=200` for one configuration and the next configuration in the
same session silently inherited it, which made a truncating baseline look
untruncated. These tests pin the fix (brief A12b).

They need a populated database, so they skip when one is not available — CI runs
Postgres but ingests no corpus.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select, text

from papertrail.database import session_scope
from papertrail.models import Chunk
from papertrail.repository import vector_search

MIN_EMBEDDED_CHUNKS = 250
PROBE = [1.0] + [0.0] * 383


def _embedded_chunk_count() -> int:
    with session_scope() as session:
        return int(
            session.scalar(
                select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
            )
            or 0
        )


try:
    _AVAILABLE = _embedded_chunk_count() >= MIN_EMBEDDED_CHUNKS
    _REASON = f"needs a database with >= {MIN_EMBEDDED_CHUNKS} embedded chunks"
except Exception as error:  # no database configured or reachable
    _AVAILABLE = False
    _REASON = f"database unavailable: {type(error).__name__}"

needs_db = pytest.mark.skipif(not _AVAILABLE, reason=_REASON)


@needs_db
def test_ef_search_does_not_leak_into_a_later_call() -> None:
    with session_scope() as session:
        baseline = len(vector_search(session, PROBE, limit=50))
        widened = len(vector_search(session, PROBE, limit=50, ef_search=200))
        after = len(vector_search(session, PROBE, limit=50))
        setting = session.execute(text("SHOW hnsw.ef_search")).scalar()

    # The default caps HNSW at 40 rows however many are asked for; widening lifts it.
    assert baseline == 40
    assert widened == 50
    # The whole point: the third call must not inherit ef_search=200.
    assert after == baseline
    assert setting == "40"


@needs_db
def test_candidate_limit_alone_cannot_exceed_the_ef_cap() -> None:
    with session_scope() as session:
        assert len(vector_search(session, PROBE, limit=200)) == 40


@needs_db
def test_exact_does_not_leak_into_a_later_approximate_call() -> None:
    with session_scope() as session:
        exact_rows = len(vector_search(session, PROBE, limit=50, exact=True))
        approx_rows = len(vector_search(session, PROBE, limit=50))
        index_scan_enabled = session.execute(text("SHOW enable_indexscan")).scalar()
        bitmap_enabled = session.execute(text("SHOW enable_bitmapscan")).scalar()

    # exact=True disables the index, so a sequential scan returns all 50 asked for.
    assert exact_rows == 50
    # The following approximate call must be back on the index, hence capped at 40.
    assert approx_rows == 40
    assert index_scan_enabled == "on"
    assert bitmap_enabled == "on"
