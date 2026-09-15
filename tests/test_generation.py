import pytest
import httpx

import papertrail.generation as generation
from papertrail.generation import answer_question, cited_arxiv_ids


class FakeEncoder:
    model_name = "test/minilm"
    dimensions = 384

    def encode(self, texts, *, batch_size=32):
        return [[1.0] + [0.0] * 383 for _ in texts]


class FakeGenerator:
    model_name = "test/generator"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, *, question: str, context: str) -> str:
        self.calls += 1
        assert "SOURCE [2401.01234v2]" in context
        return self.answer


def retrieved(score: float = 0.03):
    return [
        {
            "arxiv_id": "2401.01234v2",
            "title": "Grounded retrieval",
            "source_url": "https://arxiv.org/abs/2401.01234v2",
            "chunk_id": 1,
            "ordinal": 0,
            "text": "The paper evaluates grounded retrieval.",
            "score": score,
        }
    ]


def test_citation_parser_deduplicates_versioned_ids() -> None:
    assert cited_arxiv_ids("A [2401.01234v2]. B [2401.01234v2].") == (
        "2401.01234v2",
    )


def test_answer_abstains_before_generation_below_threshold(monkeypatch) -> None:
    monkeypatch.setattr(generation, "retrieve", lambda *a, **k: retrieved(0.02))
    generator = FakeGenerator("Unused [2401.01234v2]")
    result = answer_question(
        object(),
        "question",
        encoder=FakeEncoder(),
        generator=generator,
        threshold=0.025,
    )
    assert result.abstained is True
    assert result.answer is None
    assert generator.calls == 0


def retrieved_three(score: float = 0.03):
    return [
        {
            "arxiv_id": f"2401.0000{i}v1",
            "title": f"Paper {i}",
            "source_url": f"https://arxiv.org/abs/2401.0000{i}v1",
            "chunk_id": i,
            "ordinal": 0,
            "text": f"excerpt {i}",
            "score": score - (i - 1) * 0.001,
        }
        for i in (1, 2, 3)
    ]


def test_soft_abstention_returns_top3_nearest_papers(monkeypatch) -> None:
    monkeypatch.setattr(generation, "retrieve", lambda *a, **k: retrieved_three(0.02))
    generator = FakeGenerator("unused")
    result = answer_question(
        object(), "question", encoder=FakeEncoder(), generator=generator, threshold=0.025,
    )
    assert result.abstained is True
    assert result.answer is None
    assert generator.calls == 0  # no generation call on abstention
    assert len(result.nearest_papers) == 3
    assert set(result.nearest_papers[0]) == {"arxiv_id", "title", "source_url"}
    assert result.gate_signal_name == "rrf_top"
    assert result.gate_score == pytest.approx(0.02)


def test_non_abstention_has_no_nearest_papers(monkeypatch) -> None:
    monkeypatch.setattr(generation, "retrieve", lambda *a, **k: retrieved())
    generator = FakeGenerator("The method evaluates retrieval [2401.01234v2].")
    result = answer_question(
        object(), "question", encoder=FakeEncoder(), generator=generator, threshold=0.025,
    )
    assert result.abstained is False
    assert result.nearest_papers == ()


def test_answer_accepts_only_retrieved_citations(monkeypatch) -> None:
    monkeypatch.setattr(generation, "retrieve", lambda *a, **k: retrieved())
    generator = FakeGenerator("The method evaluates retrieval [2401.01234v2].")
    result = answer_question(
        object(),
        "question",
        encoder=FakeEncoder(),
        generator=generator,
        threshold=0.025,
    )
    assert result.abstained is False
    assert result.citations == ("2401.01234v2",)
    assert result.retrieved_arxiv_ids == ("2401.01234v2",)


def test_answer_rejects_missing_or_invented_citations(monkeypatch) -> None:
    monkeypatch.setattr(generation, "retrieve", lambda *a, **k: retrieved())
    with pytest.raises(ValueError, match="no versioned"):
        answer_question(
            object(),
            "question",
            encoder=FakeEncoder(),
            generator=FakeGenerator("An uncited answer."),
            threshold=0.025,
        )
    with pytest.raises(ValueError, match="unsupported citations"):
        answer_question(
            object(),
            "question",
            encoder=FakeEncoder(),
            generator=FakeGenerator("An answer [9999.99999v1]."),
            threshold=0.025,
        )


def test_abstain_threshold_is_validated(monkeypatch) -> None:
    monkeypatch.setattr(generation, "retrieve", lambda *a, **k: retrieved())
    # The [0,1] check is now per-signal: rrf_top is bounded, so 1.5 is out of range.
    with pytest.raises(ValueError, match="above the valid range"):
        answer_question(
            object(),
            "question",
            encoder=FakeEncoder(),
            generator=FakeGenerator("Answer [2401.01234v2]."),
            threshold=1.5,
        )


def test_rrf_gate_over_nonhybrid_mode_recomputes_from_hybrid(monkeypatch) -> None:
    # With retrieval_mode="hybrid_rerank", hits[0]["score"] is a cross-encoder
    # logit, not an RRF score. The rrf_top gate must not read that logit; it must
    # recompute rrf_top from an independent hybrid retrieval (F4).
    import papertrail.gate as gate_module

    rerank_hits = retrieved(score=7.63)  # a cross-encoder-scale logit
    monkeypatch.setattr(generation, "retrieve", lambda *a, **k: rerank_hits)
    seen = {}

    def fake_gate_signal(session, query, encoder, name, **kwargs):
        seen["name"] = name
        return 0.05  # the true hybrid rrf_top score

    monkeypatch.setattr(gate_module, "gate_signal", fake_gate_signal)
    result = answer_question(
        object(),
        "question",
        encoder=FakeEncoder(),
        generator=FakeGenerator("Answer [2401.01234v2]."),
        threshold=0.0324,
        retrieval_mode="hybrid_rerank",
        gate_signal_name="rrf_top",
    )
    assert seen["name"] == "rrf_top"  # recomputed via gate_signal, not the logit
    assert result.gate_score == pytest.approx(0.05)  # not 7.63
    assert result.abstained is False


def test_embedding_column_reaches_every_vector_read(monkeypatch) -> None:
    # Part H trap: a non-default encoder must never be compared against the
    # MiniLM column. answer_question and every gate_signal retrieval path must
    # forward the column they were given, including the direct vector_search.
    import papertrail.gate as gate_module

    columns = []

    def fake_retrieve(*args, **kwargs):
        columns.append(("retrieve", kwargs.get("embedding_column")))
        return retrieved(score=7.63)

    def fake_ready(session, *, model_name, dimensions, column):
        columns.append(("ready", column))

    def fake_vector_search(session, embedding, *, limit, column):
        columns.append(("vector_search", column))
        return retrieved(score=0.9)

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)
    monkeypatch.setattr(gate_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(gate_module, "require_vector_search_ready", fake_ready)
    monkeypatch.setattr(gate_module, "vector_search", fake_vector_search)

    answer_question(
        object(),
        "question",
        encoder=FakeEncoder(),
        generator=FakeGenerator("Answer [2401.01234v2]."),
        threshold=0.0,
        retrieval_mode="hybrid_rerank",
        gate_signal_name="cos_top",
        embedding_column="embedding_bge_base",
    )
    assert ("vector_search", "embedding_bge_base") in columns
    assert ("ready", "embedding_bge_base") in columns
    assert all(column == "embedding_bge_base" for _, column in columns), columns


def test_groq_retries_rate_limits_without_exposing_key(monkeypatch) -> None:
    requests = []
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    responses = [
        httpx.Response(429, headers={"retry-after": "0"}, request=request),
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Answer [2401.01234v2]."}}]},
            request=request,
        ),
    ]

    def fake_post(url, **kwargs):
        requests.append((url, kwargs))
        return responses.pop(0)

    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    monkeypatch.setattr(generation.httpx, "post", fake_post)
    monkeypatch.setattr(generation.time, "sleep", lambda seconds: None)
    # Force the env fallback so the test is hermetic even when a real key is
    # configured in .env (settings.groq_api_key is preferred otherwise).
    from papertrail.config import Settings

    generator = generation.GroqGenerator(Settings(groq_api_key=None))

    assert generator.generate(question="q", context="c").startswith("Answer")
    assert len(requests) == 2
    assert requests[0][1]["headers"]["Authorization"] == "Bearer test-secret"
