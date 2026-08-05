import pytest

from papertrail.arxiv import ArxivPaper
from papertrail.chunking import chunk_text, chunks_for_paper


def test_chunking_is_deterministic_bounded_and_overlapping() -> None:
    text = " ".join(f"token{index}" for index in range(300))
    first = chunk_text(text, max_chars=240, overlap_chars=40)
    second = chunk_text(text, max_chars=240, overlap_chars=40)
    assert first == second
    assert len(first) > 1
    assert all(len(chunk) <= 240 for chunk in first)
    assert set(first[0].split()[-3:]) & set(first[1].split()[:8])


def test_paper_chunks_include_title_and_ordinals(sample_paper: ArxivPaper) -> None:
    chunks = chunks_for_paper(sample_paper)
    assert chunks[0].ordinal == 0
    assert chunks[0].text.startswith("A Paper Title\n\n")


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", max_chars=100, overlap_chars=100)


@pytest.fixture
def sample_paper() -> ArxivPaper:
    from datetime import UTC, datetime

    timestamp = datetime(2026, 8, 5, tzinfo=UTC)
    return ArxivPaper(
        arxiv_id="2608.00001v1",
        title="A Paper Title",
        abstract="A real abstract about retrieval.",
        authors=("May Example",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published_at=timestamp,
        updated_at=timestamp,
        source_url="https://arxiv.org/abs/2608.00001v1",
    )

