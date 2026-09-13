"""Sentence-transformer encoding behind a small, testable interface."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from .config import get_settings
from .models import EMBEDDING_DIMENSIONS


class Encoder(Protocol):
    model_name: str
    dimensions: int

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]:
        """Return one normalized embedding per input text, as a QUERY."""


class SentenceTransformerEncoder:
    """A sentence-transformer with explicit dimensions and asymmetric prefixes.

    Several retrieval encoders are asymmetric: e5 wants ``query: ``/``passage: ``,
    bge wants a query instruction and nothing on passages. Applying a prefix at
    index time but not at query time (or the reverse) quietly wrecks retrieval, so
    the two paths are separate and named:

    * ``encode()`` is the **query** path and applies ``query_prefix``. It is the
      default because almost every caller — ``retrieve``, the gate, the study
      scripts — encodes queries.
    * ``encode_passage()`` is the **indexing** path and applies ``passage_prefix``.
      ``pipeline.embed_manifest_corpus`` is the only caller.

    With both prefixes empty (MiniLM) the two are identical and behaviour is exactly
    as before, so the frozen evidence still reproduces. Note that
    ``SentenceTransformer.prompts`` is an empty default for every model used here,
    so the library applies no prefix of its own; these are the only prefixes.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        dimensions: int | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required; install PaperTrail with the ml extra"
            ) from error
        self.model_name = model_name or get_settings().embedding_model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model = SentenceTransformer(self.model_name)
        dimension_method = getattr(self._model, "get_embedding_dimension", None)
        if dimension_method is None:
            dimension_method = self._model.get_sentence_embedding_dimension
        self.dimensions = int(dimension_method())
        # The dimension check stays; the expectation is now a constructor argument
        # so a 768-d model can be used against its own column (brief H1).
        expected = EMBEDDING_DIMENSIONS if dimensions is None else int(dimensions)
        if self.dimensions != expected:
            raise ValueError(
                f"encoder {self.model_name} emits {self.dimensions} dimensions; "
                f"expected {expected}"
            )

    def _encode(self, texts: Sequence[str], prefix: str, batch_size: int) -> list[list[float]]:
        if not texts:
            return []
        prepared = [f"{prefix}{text}" for text in texts] if prefix else list(texts)
        vectors = self._model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result = vectors.tolist()
        validate_embeddings(result, expected_count=len(texts), dimensions=self.dimensions)
        return result

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]:
        """Encode as queries (applies ``query_prefix``)."""
        return self._encode(texts, self.query_prefix, batch_size)

    def encode_passage(self, texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]:
        """Encode as passages for indexing (applies ``passage_prefix``)."""
        return self._encode(texts, self.passage_prefix, batch_size)


def validate_embeddings(
    vectors: Sequence[Sequence[float]], *, expected_count: int, dimensions: int
) -> None:
    if len(vectors) != expected_count:
        raise ValueError(
            f"encoder returned {len(vectors)} vectors for {expected_count} texts"
        )
    invalid = [index for index, vector in enumerate(vectors) if len(vector) != dimensions]
    if invalid:
        raise ValueError(
            f"encoder returned wrong dimensions at rows {invalid[:5]}; expected {dimensions}"
        )


@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder()
