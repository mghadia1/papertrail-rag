from pathlib import Path
from contextlib import contextmanager

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from papertrail.api import app
import papertrail.api as api_module
from papertrail.generation import AnswerResult
from papertrail.models import Chunk


def test_health_exposes_honesty_gate() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["resume_eligible"] is False
    assert response.json()["milestone"] == "evaluated-rag"


def test_ask_api_returns_grounded_answer(monkeypatch) -> None:
    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr(api_module, "session_scope", fake_session_scope)
    monkeypatch.setattr(api_module, "get_encoder", lambda: object())
    monkeypatch.setattr(api_module, "get_generator", lambda: object())
    monkeypatch.setattr(
        api_module,
        "answer_question",
        lambda *args, **kwargs: AnswerResult(
            question="What is RRF?",
            answer="It fuses ranked lists [2401.01234v2].",
            abstained=False,
            abstain_reason=None,
            retrieval_mode="hybrid",
            top_score=0.04,
            threshold=0.03239446668849102,
            generator_model="test/model",
            citations=("2401.01234v2",),
            retrieved_arxiv_ids=("2401.01234v2",),
        ),
    )
    response = TestClient(app).get("/ask", params={"q": "What is RRF?"})
    assert response.status_code == 200
    assert response.json()["citations"] == ["2401.01234v2"]


def test_schema_uses_384_dimension_vectors() -> None:
    embedding = Chunk.__table__.c.embedding.type
    assert embedding.dim == 384


def test_postgres_indexes_compile_to_hnsw_cosine_and_gin() -> None:
    statements = [
        str(index.compile(dialect=postgresql.dialect()))
        if hasattr(index, "compile")
        else str(index)
        for index in Chunk.__table__.indexes
    ]
    # SQLAlchemy Index objects compile through CreateIndex.
    from sqlalchemy.schema import CreateIndex

    sql = "\n".join(
        str(CreateIndex(index).compile(dialect=postgresql.dialect()))
        for index in Chunk.__table__.indexes
    )
    assert "USING hnsw" in sql
    assert "vector_cosine_ops" in sql
    assert "USING gin" in sql
    assert statements


def test_migration_enables_vector_and_both_indexes() -> None:
    migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "20260805_0001_create_papers_and_chunks.py"
    ).read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in migration
    assert "USING hnsw" in migration
    assert "vector_cosine_ops" in migration
    assert "USING gin" in migration
