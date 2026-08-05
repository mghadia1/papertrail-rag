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
        """Return one normalized embedding per input text."""


class SentenceTransformerEncoder:
    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required; install PaperTrail with the ml extra"
            ) from error
        self.model_name = model_name or get_settings().embedding_model
        self._model = SentenceTransformer(self.model_name)
        dimension_method = getattr(self._model, "get_embedding_dimension", None)
        if dimension_method is None:
            dimension_method = self._model.get_sentence_embedding_dimension
        self.dimensions = int(dimension_method())
        if self.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"encoder {self.model_name} emits {self.dimensions} dimensions; "
                f"database requires {EMBEDDING_DIMENSIONS}"
            )

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result = vectors.tolist()
        validate_embeddings(result, expected_count=len(texts), dimensions=self.dimensions)
        return result


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
