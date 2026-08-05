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
    with pytest.raises(ValueError, match="between 0 and 1"):
        answer_question(
            object(),
            "question",
            encoder=FakeEncoder(),
            generator=FakeGenerator("Answer [2401.01234v2]."),
            threshold=1.5,
        )


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
    generator = generation.GroqGenerator()

    assert generator.generate(question="q", context="c").startswith("Answer")
    assert len(requests) == 2
    assert requests[0][1]["headers"]["Authorization"] == "Bearer test-secret"
