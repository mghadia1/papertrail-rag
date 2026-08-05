"""Deterministic text chunking used before embedding and indexing."""

from __future__ import annotations

from dataclasses import dataclass

from .arxiv import ArxivPaper, normalize_whitespace


@dataclass(frozen=True)
class PaperChunk:
    ordinal: int
    text: str


def chunk_text(text: str, *, max_chars: int = 1000, overlap_chars: int = 150) -> list[str]:
    """Split normalized text near word boundaries with deterministic overlap."""

    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if not chunk:
            raise RuntimeError("chunking made no progress")
        chunks.append(chunk)
        if end == len(normalized):
            break
        next_start = max(end - overlap_chars, start + 1)
        while next_start < end and normalized[next_start] != " ":
            next_start += 1
        start = min(next_start + 1, end)
    return chunks


def chunks_for_paper(paper: ArxivPaper) -> list[PaperChunk]:
    abstract_chunks = chunk_text(paper.abstract)
    return [
        PaperChunk(ordinal=index, text=f"{paper.title}\n\n{abstract}")
        for index, abstract in enumerate(abstract_chunks)
    ]

