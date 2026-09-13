"""Tests for asymmetric encoder prefixes and embedding-column selection (Part H).

The failure this guards against is the brief's main Phase 5 trap: applying a prefix
at index time but not at query time, or the reverse. It is silent — retrieval simply
gets worse — so the query and passage paths are separate methods and these tests pin
which prefix each one applies.
"""

from __future__ import annotations

import pytest

from papertrail.embedding import SentenceTransformerEncoder, validate_embeddings
from papertrail.models import EMBEDDING_COLUMNS
from papertrail.repository import embedding_column


class _FakeST:
    """Stands in for SentenceTransformer, recording the texts it was handed."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.seen: list[list[str]] = []

    def get_embedding_dimension(self) -> int:
        return self.dimensions

    def encode(self, texts, **kwargs):
        self.seen.append(list(texts))
        # The encoder calls .tolist() on whatever the model returns. Returning a
        # stub rather than a numpy array keeps this test hermetic: numpy only
        # arrives with the `ml` extra, and CI installs `.[dev]` alone.
        return _ArrayStub([[0.0] * self.dimensions for _ in texts])


class _ArrayStub:
    """Minimal stand-in for the numpy array SentenceTransformer.encode returns."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


def _encoder(monkeypatch, dimensions=384, **kwargs) -> tuple[SentenceTransformerEncoder, _FakeST]:
    fake = _FakeST(dimensions)
    encoder = SentenceTransformerEncoder.__new__(SentenceTransformerEncoder)
    encoder.model_name = "fake/model"
    encoder.query_prefix = kwargs.get("query_prefix", "")
    encoder.passage_prefix = kwargs.get("passage_prefix", "")
    encoder._model = fake
    encoder.dimensions = dimensions
    return encoder, fake


def test_encode_applies_the_query_prefix_and_encode_passage_the_passage_prefix(monkeypatch) -> None:
    encoder, fake = _encoder(monkeypatch, query_prefix="query: ", passage_prefix="passage: ")
    encoder.encode(["how do transformers work"])
    encoder.encode_passage(["Attention is all you need"])
    assert fake.seen[0] == ["query: how do transformers work"]
    assert fake.seen[1] == ["passage: Attention is all you need"]


def test_empty_prefixes_leave_text_untouched(monkeypatch) -> None:
    # MiniLM's case: both paths must be byte-identical to the pre-Phase-5 behaviour,
    # which is what keeps the frozen evidence reproducible.
    encoder, fake = _encoder(monkeypatch)
    encoder.encode(["a query"])
    encoder.encode_passage(["a passage"])
    assert fake.seen == [["a query"], ["a passage"]]


def test_asymmetric_model_may_have_a_query_prefix_and_no_passage_prefix(monkeypatch) -> None:
    # bge's documented shape: instruction on the query, nothing on passages.
    instruction = "Represent this sentence for searching relevant passages: "
    encoder, fake = _encoder(monkeypatch, query_prefix=instruction)
    encoder.encode(["sparse retrieval"])
    encoder.encode_passage(["Some abstract text"])
    assert fake.seen[0] == [instruction + "sparse retrieval"]
    assert fake.seen[1] == ["Some abstract text"]


def test_encode_returns_empty_for_no_texts(monkeypatch) -> None:
    encoder, _ = _encoder(monkeypatch)
    assert encoder.encode([]) == []
    assert encoder.encode_passage([]) == []


def test_embedding_column_whitelist_rejects_unknown_names() -> None:
    assert "embedding" in EMBEDDING_COLUMNS
    assert EMBEDDING_COLUMNS["embedding_bge_base"] == 768
    # A column name never reaches SQL unchecked.
    for bad in ["embedding; DROP TABLE chunks", "text", "", "embedding_nope"]:
        with pytest.raises(ValueError, match="unknown embedding column"):
            embedding_column(bad)
    assert embedding_column("embedding_bge_small") is not None


def test_validate_embeddings_catches_a_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="wrong dimensions"):
        validate_embeddings([[0.0] * 384], expected_count=1, dimensions=768)
