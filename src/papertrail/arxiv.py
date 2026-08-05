"""Small, polite client for the public arXiv Atom API."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import httpx

from .config import Settings, get_settings


ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"
DEFAULT_CATEGORIES = ("cs.LG", "cs.CL", "cs.CV")
MAX_PAGE_SIZE = 2000
MAX_ID_BATCH_SIZE = 100


@dataclass(frozen=True)
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    primary_category: str
    published_at: datetime
    updated_at: datetime
    source_url: str


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def category_query(categories: Iterable[str]) -> str:
    values = tuple(dict.fromkeys(category.strip() for category in categories if category.strip()))
    if not values:
        raise ValueError("at least one arXiv category is required")
    if any(not re.fullmatch(r"[a-z-]+\.[A-Z]+", value) for value in values):
        raise ValueError(f"invalid arXiv category list: {values}")
    return " OR ".join(f"cat:{value}" for value in values)


def _required_text(node: ET.Element, path: str, label: str) -> str:
    child = node.find(path)
    if child is None or not child.text or not child.text.strip():
        raise ValueError(f"arXiv entry is missing {label}")
    return child.text.strip()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_atom_feed(payload: str | bytes) -> list[ArxivPaper]:
    """Parse the fields PaperTrail persists from an arXiv Atom response."""

    root = ET.fromstring(payload)
    papers: list[ArxivPaper] = []
    for entry in root.findall(f"{{{ATOM}}}entry"):
        entry_url = _required_text(entry, f"{{{ATOM}}}id", "id")
        marker = "/abs/"
        if marker not in entry_url:
            raise ValueError(f"unexpected arXiv entry id: {entry_url}")
        arxiv_id = entry_url.split(marker, 1)[1].strip("/")
        authors = tuple(
            normalize_whitespace(_required_text(author, f"{{{ATOM}}}name", "author name"))
            for author in entry.findall(f"{{{ATOM}}}author")
        )
        categories = tuple(
            category.attrib["term"]
            for category in entry.findall(f"{{{ATOM}}}category")
            if category.attrib.get("term")
        )
        primary = entry.find(f"{{{ARXIV}}}primary_category")
        primary_category = (
            primary.attrib.get("term", "") if primary is not None else ""
        )
        if not primary_category:
            primary_category = categories[0] if categories else "unknown"
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=normalize_whitespace(
                    _required_text(entry, f"{{{ATOM}}}title", "title")
                ),
                abstract=normalize_whitespace(
                    _required_text(entry, f"{{{ATOM}}}summary", "abstract")
                ),
                authors=authors,
                categories=categories,
                primary_category=primary_category,
                published_at=_parse_datetime(
                    _required_text(entry, f"{{{ATOM}}}published", "published date")
                ),
                updated_at=_parse_datetime(
                    _required_text(entry, f"{{{ATOM}}}updated", "updated date")
                ),
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    return papers


def fetch_papers(
    *,
    max_results: int = 20,
    start: int = 0,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> list[ArxivPaper]:
    """Fetch one arXiv page; callers doing pagination must wait three seconds."""

    if not 1 <= max_results <= MAX_PAGE_SIZE:
        raise ValueError(f"max_results must be between 1 and {MAX_PAGE_SIZE}")
    if start < 0:
        raise ValueError("start must be non-negative")
    config = settings or get_settings()
    params = {
        "search_query": category_query(categories),
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    owns_client = client is None
    http_client = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        response = http_client.get(
            config.arxiv_api_url,
            params=params,
            headers={"User-Agent": config.arxiv_user_agent},
        )
        response.raise_for_status()
        papers = parse_atom_feed(response.content)
    finally:
        if owns_client:
            http_client.close()
    if len(papers) > max_results:
        raise ValueError("arXiv returned more entries than requested")
    return papers


def fetch_papers_by_ids(
    arxiv_ids: Iterable[str],
    *,
    batch_size: int = MAX_ID_BATCH_SIZE,
    request_delay_seconds: float = 3.0,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> list[ArxivPaper]:
    """Fetch an exact versioned-ID corpus in polite, validated batches."""

    raw_requested = tuple(value.strip() for value in arxiv_ids if value.strip())
    requested = tuple(dict.fromkeys(raw_requested))
    if not requested:
        raise ValueError("at least one arXiv ID is required")
    if len(requested) != len(raw_requested):
        raise ValueError("arXiv ID replay request contains duplicates")
    identifier_pattern = re.compile(r"[A-Za-z0-9.-]+(?:/[A-Za-z0-9.-]+)?v\d+")
    invalid = [value for value in requested if not identifier_pattern.fullmatch(value)]
    if invalid:
        raise ValueError(f"invalid versioned arXiv IDs: {invalid[:3]}")
    if not 1 <= batch_size <= MAX_ID_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_ID_BATCH_SIZE}")
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must be non-negative")

    config = settings or get_settings()
    owns_client = client is None
    http_client = client or httpx.Client(timeout=30.0, follow_redirects=True)
    fetched: dict[str, ArxivPaper] = {}
    try:
        for offset in range(0, len(requested), batch_size):
            batch = requested[offset : offset + batch_size]
            response = http_client.get(
                config.arxiv_api_url,
                params={"id_list": ",".join(batch), "max_results": len(batch)},
                headers={"User-Agent": config.arxiv_user_agent},
            )
            response.raise_for_status()
            for paper in parse_atom_feed(response.content):
                if paper.arxiv_id in fetched:
                    raise ValueError(f"arXiv replay returned duplicate ID {paper.arxiv_id}")
                fetched[paper.arxiv_id] = paper
            if offset + batch_size < len(requested) and request_delay_seconds:
                time.sleep(request_delay_seconds)
    finally:
        if owns_client:
            http_client.close()

    missing = sorted(set(requested) - set(fetched))
    unexpected = sorted(set(fetched) - set(requested))
    if missing or unexpected:
        raise ValueError(
            "arXiv manifest replay mismatch: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
        )
    return [fetched[identifier] for identifier in requested]
