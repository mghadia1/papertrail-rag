"""Command-line entry points for proving the arXiv path and ingesting metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arxiv import DEFAULT_CATEGORIES, fetch_papers, fetch_papers_by_ids
from .database import session_scope
from .config import get_settings
from .embedding import get_encoder
from .evaluation import evaluate, evaluate_rag, load_question_set
from .evidence import verify_rag_evidence, verify_retrieval_evidence
from .generation import answer_question, get_generator
from .manifest import CorpusManifest
from .pipeline import embed_manifest_corpus
from .repository import (
    ingest_papers,
    stored_arxiv_ids,
)
from .retrieval import retrieve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaperTrail arXiv ingestion tools")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="fetch and print real arXiv metadata")
    fetch.add_argument("--limit", type=int, default=20)
    fetch.add_argument("--category", action="append", dest="categories")
    ingest = commands.add_parser("ingest", help="fetch arXiv metadata and upsert Postgres")
    ingest.add_argument("--limit", type=int, default=20)
    ingest.add_argument("--category", action="append", dest="categories")
    export = commands.add_parser(
        "export-manifest", help="export the exact stored arXiv ID set"
    )
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--category", action="append", dest="categories")
    replay = commands.add_parser(
        "replay-manifest", help="fetch and ingest exactly the IDs in a manifest"
    )
    replay.add_argument("manifest", type=Path)
    replay.add_argument("--batch-size", type=int, default=100)
    replay.add_argument("--request-delay", type=float, default=3.0)
    verify = commands.add_parser(
        "verify-manifest", help="compare the stored corpus with an exact manifest"
    )
    verify.add_argument("manifest", type=Path)
    embed = commands.add_parser(
        "embed", help="embed every chunk in an exact corpus manifest"
    )
    embed.add_argument("manifest", type=Path)
    embed.add_argument("--batch-size", type=int, default=32)
    search = commands.add_parser("search", help="search embedded paper chunks")
    search.add_argument("query")
    search.add_argument(
        "--mode",
        choices=("vector", "keyword", "hybrid", "hybrid_rerank"),
        default="hybrid",
    )
    search.add_argument("--limit", type=int, default=5)
    ask = commands.add_parser("ask", help="answer from hybrid-retrieved paper chunks")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    evaluation = commands.add_parser(
        "evaluate", help="run the frozen retrieval and abstention evaluation"
    )
    evaluation.add_argument("--questions", type=Path, required=True)
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    rag_eval = commands.add_parser(
        "evaluate-rag", help="run held-out generation, abstention, and citation checks"
    )
    rag_eval.add_argument("--questions", type=Path, required=True)
    rag_eval.add_argument("--manifest", type=Path, required=True)
    rag_eval.add_argument("--output", type=Path, required=True)
    evidence = commands.add_parser(
        "verify-evidence", help="recompute and verify a retrieval or RAG report"
    )
    evidence.add_argument("--kind", choices=("retrieval", "rag"), required=True)
    evidence.add_argument("--report", type=Path, required=True)
    evidence.add_argument("--manifest", type=Path, required=True)
    evidence.add_argument("--questions", type=Path)
    return parser


def _fetch(args: argparse.Namespace):
    categories = tuple(args.categories or DEFAULT_CATEGORIES)
    return fetch_papers(max_results=args.limit, categories=categories)


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "export-manifest":
        categories = tuple(args.categories or DEFAULT_CATEGORIES)
        with session_scope() as session:
            manifest = CorpusManifest.create(
                stored_arxiv_ids(session), categories=categories
            )
        manifest.write(args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "papers": len(manifest.arxiv_ids),
                    "arxiv_ids_sha256": manifest.arxiv_ids_sha256,
                }
            )
        )
        return 0
    if args.command == "verify-manifest":
        manifest = CorpusManifest.read(args.manifest)
        with session_scope() as session:
            actual_ids = stored_arxiv_ids(session)
        actual = set(actual_ids)
        expected = set(manifest.arxiv_ids)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise SystemExit(
                "manifest/database mismatch: "
                f"missing={missing[:5]} ({len(missing)} total), "
                f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
            )
        print(
            json.dumps(
                {
                    "verified": True,
                    "papers": len(actual_ids),
                    "arxiv_ids_sha256": manifest.arxiv_ids_sha256,
                }
            )
        )
        return 0
    if args.command == "replay-manifest":
        manifest = CorpusManifest.read(args.manifest)
        papers = fetch_papers_by_ids(
            manifest.arxiv_ids,
            batch_size=args.batch_size,
            request_delay_seconds=args.request_delay,
        )
        with session_scope() as session:
            result = ingest_papers(session, papers)
        print(
            json.dumps(
                {
                    **result,
                    "manifest": str(args.manifest),
                    "arxiv_ids_sha256": manifest.arxiv_ids_sha256,
                }
            )
        )
        return 0
    if args.command == "embed":
        manifest = CorpusManifest.read(args.manifest)
        result = embed_manifest_corpus(
            manifest, get_encoder(), batch_size=args.batch_size
        )
        print(json.dumps(result))
        return 0
    if args.command == "search":
        if not 1 <= args.limit <= 50:
            raise SystemExit("--limit must be between 1 and 50")
        encoder = get_encoder() if args.mode != "keyword" else None
        with session_scope() as session:
            results = retrieve(
                session,
                args.query,
                mode=args.mode,
                limit=args.limit,
                encoder=encoder,
            )
        print(json.dumps({"mode": args.mode, "query": args.query, "results": results}))
        return 0
    if args.command == "ask":
        settings = get_settings()
        with session_scope() as session:
            result = answer_question(
                session,
                args.question,
                encoder=get_encoder(),
                generator=get_generator(),
                threshold=settings.abstain_threshold,
                top_k=args.top_k,
            )
        print(json.dumps(result.__dict__))
        return 0
    if args.command == "evaluate":
        manifest = CorpusManifest.read(args.manifest)
        question_set = load_question_set(args.questions, manifest)
        with session_scope() as session:
            report = evaluate(
                session,
                question_set=question_set,
                manifest=manifest,
                encoder=get_encoder(),
                output_path=args.output,
            )
        print(json.dumps({"output": str(args.output), **report["aggregates"], "abstention": report["abstention"]}))
        return 0
    if args.command == "evaluate-rag":
        manifest = CorpusManifest.read(args.manifest)
        question_set = load_question_set(args.questions, manifest)
        settings = get_settings()
        with session_scope() as session:
            report = evaluate_rag(
                session,
                question_set=question_set,
                encoder=get_encoder(),
                generator=get_generator(),
                threshold=settings.abstain_threshold,
                output_path=args.output,
            )
        print(json.dumps({key: value for key, value in report.items() if key != "records"}))
        return 0
    if args.command == "verify-evidence":
        manifest = CorpusManifest.read(args.manifest)
        if args.kind == "retrieval":
            question_set = (
                load_question_set(args.questions, manifest)
                if args.questions is not None
                else None
            )
            result = verify_retrieval_evidence(
                args.report, manifest, question_set=question_set
            )
        else:
            if args.questions is None:
                raise SystemExit("--questions is required for RAG evidence")
            question_set = load_question_set(args.questions, manifest)
            result = verify_rag_evidence(
                args.report,
                question_set=question_set,
                expected_threshold=get_settings().abstain_threshold,
            )
        print(json.dumps(result))
        return 0
    papers = _fetch(args)
    if args.command == "fetch":
        for paper in papers:
            print(
                json.dumps(
                    {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "primary_category": paper.primary_category,
                        "source_url": paper.source_url,
                    }
                )
            )
        print(json.dumps({"fetched": len(papers), "stored": 0}))
        return 0
    with session_scope() as session:
        result = ingest_papers(session, papers)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
