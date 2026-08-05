"""Exact corpus manifests for reproducible arXiv ingestion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = 1


def canonical_ids(ids: Iterable[str]) -> tuple[str, ...]:
    values = tuple(sorted(identifier.strip() for identifier in ids if identifier.strip()))
    if not values:
        raise ValueError("corpus manifest must contain at least one arXiv ID")
    if len(values) != len(set(values)):
        raise ValueError("corpus manifest contains duplicate arXiv IDs")
    return values


def ids_sha256(ids: Iterable[str]) -> str:
    values = canonical_ids(ids)
    payload = "".join(f"{identifier}\n" for identifier in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CorpusManifest:
    created_at_utc: str
    categories: tuple[str, ...]
    arxiv_ids: tuple[str, ...]
    arxiv_ids_sha256: str
    schema_version: int = MANIFEST_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        ids: Iterable[str],
        *,
        categories: Iterable[str],
        created_at: datetime | None = None,
    ) -> "CorpusManifest":
        ordered_ids = canonical_ids(ids)
        category_values = tuple(dict.fromkeys(value.strip() for value in categories if value.strip()))
        if not category_values:
            raise ValueError("corpus manifest must record at least one category")
        timestamp = created_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("manifest timestamp must be timezone-aware")
        return cls(
            created_at_utc=timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            categories=category_values,
            arxiv_ids=ordered_ids,
            arxiv_ids_sha256=ids_sha256(ordered_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "source": "arXiv Atom API",
            "categories": list(self.categories),
            "paper_count": len(self.arxiv_ids),
            "arxiv_ids_sha256": self.arxiv_ids_sha256,
            "hash_input": "arXiv IDs sorted ascending, one UTF-8 ID per line with a trailing newline",
            "arxiv_ids": list(self.arxiv_ids),
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "CorpusManifest":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read corpus manifest {path}: {error}") from error
        if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema_version: {payload.get('schema_version')}"
            )
        ids = canonical_ids(payload.get("arxiv_ids", ()))
        if payload.get("paper_count") != len(ids):
            raise ValueError(
                "manifest paper_count does not match arxiv_ids: "
                f"{payload.get('paper_count')} != {len(ids)}"
            )
        published_digest = payload.get("arxiv_ids_sha256")
        actual_digest = ids_sha256(ids)
        if published_digest != actual_digest:
            raise ValueError(
                "manifest arxiv_ids_sha256 mismatch: "
                f"published {published_digest}, recomputed {actual_digest}"
            )
        categories = tuple(payload.get("categories", ()))
        if not categories or any(not isinstance(value, str) or not value for value in categories):
            raise ValueError("manifest categories must be a non-empty string list")
        created_at = payload.get("created_at_utc")
        if not isinstance(created_at, str) or not created_at.endswith("Z"):
            raise ValueError("manifest created_at_utc must be an ISO UTC timestamp ending in Z")
        return cls(
            created_at_utc=created_at,
            categories=categories,
            arxiv_ids=ids,
            arxiv_ids_sha256=actual_digest,
        )
