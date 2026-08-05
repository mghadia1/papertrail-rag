import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from papertrail.manifest import CorpusManifest, ids_sha256


def test_manifest_is_sorted_hashed_and_round_trips(tmp_path: Path) -> None:
    manifest = CorpusManifest.create(
        ["2402.00002v1", "2401.00001v2"],
        categories=["cs.LG", "cs.CL"],
        created_at=datetime(2026, 8, 5, 20, 0, tzinfo=UTC),
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)

    loaded = CorpusManifest.read(path)
    assert loaded == manifest
    assert loaded.arxiv_ids == ("2401.00001v2", "2402.00002v1")
    assert loaded.arxiv_ids_sha256 == ids_sha256(loaded.arxiv_ids)


def test_manifest_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        CorpusManifest.create(
            ["2401.00001v1", "2401.00001v1"], categories=["cs.LG"]
        )


def test_manifest_recomputes_digest_instead_of_trusting_file(tmp_path: Path) -> None:
    manifest = CorpusManifest.create(["2401.00001v1"], categories=["cs.LG"])
    path = tmp_path / "manifest.json"
    manifest.write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["arxiv_ids_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        CorpusManifest.read(path)


def test_manifest_rejects_edited_count(tmp_path: Path) -> None:
    manifest = CorpusManifest.create(["2401.00001v1"], categories=["cs.LG"])
    path = tmp_path / "manifest.json"
    manifest.write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["paper_count"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="paper_count"):
        CorpusManifest.read(path)
