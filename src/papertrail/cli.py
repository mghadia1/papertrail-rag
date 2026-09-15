"""Command-line entry points for proving the arXiv path and ingesting metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arxiv import DEFAULT_CATEGORIES, fetch_papers, fetch_papers_by_ids
from .database import session_scope
from .config import get_settings
from .embedding import get_encoder
from .evaluation import evaluate, evaluate_rag, load_question_set
from .evidence import (
    verify_embed_cost_evidence,
    verify_fusion_evidence,
    verify_sparse_evidence,
    verify_gate_evidence,
    verify_hnsw_evidence,
    verify_rag_evidence,
    verify_retrieval_evidence,
)
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
    embed.add_argument("--model", default=None,
                       help="encoder to embed with (default: the configured model)")
    embed.add_argument("--column", default="embedding",
                       help="embedding column to fill (default: embedding, MiniLM's frozen column)")
    embed.add_argument("--passage-prefix", default="",
                       help="prefix applied to chunk text at index time (e.g. 'passage: ')")
    embed.add_argument("--query-prefix", default="",
                       help="query prefix this column is meant to be searched with; recorded "
                            "in the run row so evaluation can default to it")
    search = commands.add_parser("search", help="search embedded paper chunks")
    search.add_argument("query")
    search.add_argument(
        "--mode",
        choices=("vector", "keyword", "hybrid", "hybrid_rerank", "vector_rerank"),
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
    evaluation.add_argument(
        "--modes",
        nargs="+",
        choices=("vector", "keyword", "hybrid", "hybrid_rerank", "vector_rerank"),
        default=None,
        help="subset of modes to evaluate (default: all modes for the schema)",
    )
    evaluation.add_argument(
        "--reranker",
        default=None,
        help="cross-encoder model name for rerank modes (default: ms-marco-MiniLM-L-6-v2)",
    )
    evaluation.add_argument(
        "--rerank-pool",
        type=int,
        default=None,
        help="first-stage candidate pool depth for rerank modes (default: max(limit*2, 20))",
    )
    evaluation.add_argument(
        "--reranker-max-length",
        type=int,
        default=None,
        help="explicit CrossEncoder max_length (bge-reranker-base needs 512)",
    )
    evaluation.add_argument("--model", default=None,
                            help="query encoder (default: the configured model)")
    evaluation.add_argument("--column", default="embedding",
                            help="embedding column to search (default: embedding)")
    evaluation.add_argument(
        "--splits", nargs="+", choices=("development", "heldout"),
        default=["development", "heldout"],
        help="splits to evaluate. Use 'development' alone while comparing "
             "configurations so held-out stays reserved for one final report (A1).",
    )
    evaluation.add_argument(
        "--query-prefix", default=None,
        help="query prefix. Omit to use the prefix recorded in this column's embedding "
             "run, which is the safe default; pass it explicitly only to deliberately "
             "mismatch it, as the Phase 5 negative control does.",
    )
    rag_eval = commands.add_parser(
        "evaluate-rag", help="run held-out generation, abstention, and citation checks"
    )
    rag_eval.add_argument("--questions", type=Path, required=True)
    rag_eval.add_argument("--manifest", type=Path, required=True)
    rag_eval.add_argument("--output", type=Path, required=True)
    rag_eval.add_argument("--gate-signal", default="rrf_top")
    rag_eval.add_argument("--abstain-threshold", type=float, default=None,
                          help="override the abstain threshold (use the chosen signal's frozen dev threshold)")
    rag_eval.add_argument("--verify-entailment", action="store_true")
    evidence = commands.add_parser(
        "verify-evidence", help="recompute and verify a retrieval or RAG report"
    )
    evidence.add_argument("--kind", choices=("retrieval", "rag", "hnsw", "gate", "bm25", "sparse", "fusion", "embed-costs"), required=True)
    evidence.add_argument("--report", type=Path, required=True)
    evidence.add_argument("--manifest", type=Path, required=True)
    evidence.add_argument("--questions", type=Path)
    evidence.add_argument("--sweep", type=Path,
                          help="fusion sweep evidence; required for a held-out fusion file, which "
                               "is checked against the development-best configuration")
    evidence.add_argument("--abstain-threshold", type=float, default=None,
                          help="expected RAG abstain threshold (defaults to the config value)")
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
        if args.model is None and args.column == "embedding" and not args.passage_prefix:
            encoder = get_encoder()
        else:
            from .embedding import SentenceTransformerEncoder
            from .models import EMBEDDING_COLUMNS

            if args.column not in EMBEDDING_COLUMNS:
                raise SystemExit(
                    f"unknown column {args.column!r}; expected one of {sorted(EMBEDDING_COLUMNS)}"
                )
            encoder = SentenceTransformerEncoder(
                args.model,
                dimensions=EMBEDDING_COLUMNS[args.column],
                query_prefix=args.query_prefix,
                passage_prefix=args.passage_prefix,
            )
        result = embed_manifest_corpus(
            manifest, encoder, batch_size=args.batch_size, column=args.column
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
                gate_signal_name=settings.abstain_signal,
            )
        print(json.dumps(result.__dict__))
        if result.abstained and result.nearest_papers:
            print("Low confidence. Closest evidence:", file=sys.stderr)
            for paper in result.nearest_papers:
                print(f"  [{paper['arxiv_id']}] {paper['title']} — {paper['source_url']}", file=sys.stderr)
        return 0
    if args.command == "evaluate":
        manifest = CorpusManifest.read(args.manifest)
        question_set = load_question_set(args.questions, manifest)
        modes = tuple(args.modes) if args.modes else None
        # Query prefix defaults to whatever this column was embedded to expect, so
        # the "right prefix at index time, forgotten at query time" trap cannot
        # happen by omission. An explicit --query-prefix overrides it.
        from .models import EMBEDDING_COLUMNS

        if args.column not in EMBEDDING_COLUMNS:
            raise SystemExit(
                f"unknown column {args.column!r}; expected one of {sorted(EMBEDDING_COLUMNS)}"
            )
        query_prefix = args.query_prefix
        if query_prefix is None:
            from sqlalchemy import select as _select

            from .models import EmbeddingRun as _Run

            with session_scope() as probe:
                run = probe.scalar(
                    _select(_Run).where(_Run.column_name == args.column)
                )
                query_prefix = run.query_prefix if run is not None else ""
        if args.model is None and args.column == "embedding" and not query_prefix:
            encoder = get_encoder()
        else:
            from .embedding import SentenceTransformerEncoder

            encoder = SentenceTransformerEncoder(
                args.model,
                dimensions=EMBEDDING_COLUMNS[args.column],
                query_prefix=query_prefix,
            )
        rerank_modes = {"hybrid_rerank", "vector_rerank"}
        reranker = None
        if args.reranker is not None or (modes and any(m in rerank_modes for m in modes)):
            from .reranking import CrossEncoderReranker

            reranker = CrossEncoderReranker(
                args.reranker or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                allow_fallback=False,
                max_length=args.reranker_max_length,
            )
        with session_scope() as session:
            report = evaluate(
                session,
                question_set=question_set,
                manifest=manifest,
                encoder=encoder,
                output_path=args.output,
                modes=modes,
                reranker=reranker,
                rerank_pool=args.rerank_pool,
                embedding_column=args.column,
                splits=tuple(args.splits),
            )
        summary = {"output": str(args.output), **report["aggregates"]}
        # A split-restricted run has no abstention block (it needs both splits).
        if "abstention" in report:
            summary["abstention"] = report["abstention"]
        print(json.dumps(summary))
        return 0
    if args.command == "evaluate-rag":
        manifest = CorpusManifest.read(args.manifest)
        question_set = load_question_set(args.questions, manifest)
        settings = get_settings()
        threshold = (
            args.abstain_threshold
            if args.abstain_threshold is not None
            else settings.abstain_threshold
        )
        with session_scope() as session:
            report = evaluate_rag(
                session,
                question_set=question_set,
                encoder=get_encoder(),
                generator=get_generator(),
                threshold=threshold,
                output_path=args.output,
                gate_signal_name=args.gate_signal,
                verify_entailment=args.verify_entailment,
            )
        print(json.dumps({key: value for key, value in report.items() if key != "records"}))
        return 0
    if args.command == "verify-evidence":
        manifest = CorpusManifest.read(args.manifest)
        if args.kind == "hnsw":
            result = verify_hnsw_evidence(args.report, manifest)
        elif args.kind == "embed-costs":
            result = verify_embed_cost_evidence(args.report, manifest)
        elif args.kind == "fusion":
            question_set = (
                load_question_set(args.questions, manifest)
                if args.questions is not None
                else None
            )
            result = verify_fusion_evidence(
                args.report, manifest, question_set=question_set, sweep_path=args.sweep
            )
        elif args.kind in ("bm25", "sparse"):
            question_set = (
                load_question_set(args.questions, manifest)
                if args.questions is not None
                else None
            )
            result = verify_sparse_evidence(args.report, manifest, question_set=question_set)
        elif args.kind == "gate":
            result = verify_gate_evidence(args.report, manifest)
        elif args.kind == "retrieval":
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
            expected_threshold = (
                args.abstain_threshold
                if args.abstain_threshold is not None
                else get_settings().abstain_threshold
            )
            result = verify_rag_evidence(
                args.report,
                question_set=question_set,
                expected_threshold=expected_threshold,
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
