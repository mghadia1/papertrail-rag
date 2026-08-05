from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import papertrail.api as api_module
from papertrail.api import app
from papertrail.cli import build_parser
from papertrail.embedding import validate_embeddings
from papertrail.repository import save_embeddings


class FakeEncoder:
    model_name = "test/minilm"
    dimensions = 384

    def encode(self, texts, *, batch_size=32):
        return [[1.0] + [0.0] * 383 for _ in texts]


def test_embedding_validation_rejects_count_and_dimension_errors() -> None:
    with pytest.raises(ValueError, match="1 vectors for 2 texts"):
        validate_embeddings([[0.0] * 384], expected_count=2, dimensions=384)
    with pytest.raises(ValueError, match="wrong dimensions"):
        validate_embeddings([[0.0] * 12], expected_count=1, dimensions=384)


def test_save_embeddings_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="counts differ"):
        save_embeddings(object(), [1, 2], [[0.0] * 384])


def test_vector_search_api_returns_traceable_arxiv_hit(monkeypatch) -> None:
    @contextmanager
    def fake_scope():
        yield object()

    monkeypatch.setattr(api_module, "session_scope", fake_scope)
    monkeypatch.setattr(api_module, "get_encoder", lambda: FakeEncoder())
    monkeypatch.setattr(
        api_module,
        "retrieve",
        lambda *a, **k: [
            {
                "arxiv_id": "2401.01234v2",
                "title": "Grounded retrieval",
                "source_url": "https://arxiv.org/abs/2401.01234v2",
                "chunk_id": 7,
                "ordinal": 0,
                "text": "Grounded retrieval\n\nA test abstract.",
                "score": 0.87,
            }
        ],
    )
    response = TestClient(app).get(
        "/search", params={"q": "citation retrieval", "mode": "vector"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "test/minilm"
    assert body["results"][0]["arxiv_id"] == "2401.01234v2"
    assert body["results"][0]["source_url"].endswith("2401.01234v2")


def test_vector_search_api_rejects_unready_embeddings(monkeypatch) -> None:
    @contextmanager
    def fake_scope():
        yield object()

    def reject(*args, **kwargs):
        raise ValueError("vector search is not ready: embedded 3/10 chunks")

    monkeypatch.setattr(api_module, "session_scope", fake_scope)
    monkeypatch.setattr(api_module, "get_encoder", lambda: FakeEncoder())
    monkeypatch.setattr(api_module, "retrieve", reject)
    response = TestClient(app).get("/search", params={"q": "robot learning"})
    assert response.status_code == 409
    assert "embedded 3/10" in response.json()["detail"]


def test_cli_and_container_include_e3_surfaces() -> None:
    parser = build_parser()
    assert parser.parse_args(["embed", "manifest.json"]).command == "embed"
    assert parser.parse_args(["search", "medical robotics"]).command == "search"
    assert parser.parse_args(["search", "medical robotics"]).mode == "hybrid"
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")
    assert "'.[ml]'" in dockerfile


def test_embedding_migration_records_provenance_and_progress() -> None:
    migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "20260805_0002_add_embedding_runs.py"
    ).read_text(encoding="utf-8")
    assert "corpus_arxiv_ids_sha256" in migration
    assert "total_chunk_count" in migration
    assert "status IN ('running', 'complete')" in migration
