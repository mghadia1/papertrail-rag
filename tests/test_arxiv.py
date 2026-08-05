from pathlib import Path

import httpx
import pytest

from papertrail.arxiv import (
    category_query,
    fetch_papers,
    fetch_papers_by_ids,
    parse_atom_feed,
)
from papertrail.config import Settings


FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_feed.xml"


def test_atom_parser_preserves_verifiable_arxiv_metadata() -> None:
    papers = parse_atom_feed(FIXTURE.read_bytes())
    assert len(papers) == 2
    assert papers[0].arxiv_id == "2401.01234v2"
    assert papers[0].title == "Retrieval-Augmented Models for Testing"
    assert papers[0].authors == ("Ada Researcher", "Grace Scientist")
    assert papers[0].categories == ("cs.CL", "cs.LG")
    assert papers[0].source_url == "https://arxiv.org/abs/2401.01234v2"


def test_category_query_is_deterministic_and_validated() -> None:
    assert category_query(("cs.LG", "cs.CL", "cs.LG")) == "cat:cs.LG OR cat:cs.CL"
    with pytest.raises(ValueError, match="invalid arXiv category"):
        category_query(("machine-learning",))


def test_fetch_uses_bounded_page_and_identifying_user_agent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["max_results"] == "2"
        assert request.url.params["sortBy"] == "submittedDate"
        assert "cat:cs.LG" in request.url.params["search_query"]
        assert request.headers["user-agent"].startswith("PaperTrail/")
        return httpx.Response(200, content=FIXTURE.read_bytes())

    settings = Settings(
        arxiv_api_url="https://example.test/api/query",
        arxiv_user_agent="PaperTrail/test (github.com/mghadia1)",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        papers = fetch_papers(max_results=2, settings=settings, client=client)
    assert [paper.primary_category for paper in papers] == ["cs.CL", "cs.CV"]


@pytest.mark.parametrize("limit", [0, 2001])
def test_fetch_rejects_out_of_policy_page_sizes(limit: int) -> None:
    with pytest.raises(ValueError, match="max_results"):
        fetch_papers(max_results=limit)


def test_manifest_replay_returns_requested_order() -> None:
    requested = ("2401.09999v1", "2401.01234v2")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["id_list"] == ",".join(requested)
        return httpx.Response(200, content=FIXTURE.read_bytes())

    settings = Settings(
        arxiv_api_url="https://example.test/api/query",
        arxiv_user_agent="PaperTrail/test (github.com/mghadia1)",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        papers = fetch_papers_by_ids(
            requested,
            batch_size=2,
            request_delay_seconds=0,
            settings=settings,
            client=client,
        )
    assert [paper.arxiv_id for paper in papers] == list(requested)


def test_manifest_replay_rejects_missing_ids() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        payload = FIXTURE.read_text(encoding="utf-8")
        start = payload.index("  <entry>", payload.index("  <entry>") + 1)
        end = payload.index("  </entry>", start) + len("  </entry>\n")
        return httpx.Response(200, content=(payload[:start] + payload[end:]).encode())

    settings = Settings(
        arxiv_api_url="https://example.test/api/query",
        arxiv_user_agent="PaperTrail/test (github.com/mghadia1)",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="missing=.*2401.09999v1"):
            fetch_papers_by_ids(
                ("2401.01234v2", "2401.09999v1"),
                batch_size=2,
                request_delay_seconds=0,
                settings=settings,
                client=client,
            )


def test_manifest_replay_requires_versioned_unique_ids() -> None:
    with pytest.raises(ValueError, match="versioned"):
        fetch_papers_by_ids(("2401.01234",), request_delay_seconds=0)
    with pytest.raises(ValueError, match="duplicates"):
        fetch_papers_by_ids(
            ("2401.01234v2", "2401.01234v2"), request_delay_seconds=0
        )
