"""Measured corpus-embedding workflow with resumable provenance."""

from __future__ import annotations

from .database import session_scope
from .embedding import Encoder, validate_embeddings
from .manifest import CorpusManifest
from .repository import (
    embedding_counts,
    record_embedding_run,
    save_embeddings,
    stored_arxiv_ids,
    unembedded_chunks,
    update_embedding_run,
)


def embed_manifest_corpus(
    manifest: CorpusManifest,
    encoder: Encoder,
    *,
    batch_size: int = 32,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with session_scope() as session:
        actual_ids = stored_arxiv_ids(session)
        if actual_ids != manifest.arxiv_ids:
            missing = sorted(set(manifest.arxiv_ids) - set(actual_ids))
            unexpected = sorted(set(actual_ids) - set(manifest.arxiv_ids))
            raise ValueError(
                "manifest/database mismatch before embedding: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        embedded, total = embedding_counts(session)
        run_id, should_run = record_embedding_run(
            session,
            model_name=encoder.model_name,
            dimensions=encoder.dimensions,
            corpus_arxiv_ids_sha256=manifest.arxiv_ids_sha256,
            embedded_chunk_count=embedded,
            total_chunk_count=total,
        )

    newly_embedded = 0
    while should_run:
        with session_scope() as session:
            batch = unembedded_chunks(session, limit=batch_size)
        if not batch:
            break
        chunk_ids = [chunk_id for chunk_id, _ in batch]
        texts = [text for _, text in batch]
        vectors = encoder.encode(texts, batch_size=batch_size)
        validate_embeddings(
            vectors, expected_count=len(texts), dimensions=encoder.dimensions
        )
        with session_scope() as session:
            save_embeddings(session, chunk_ids, vectors)
            embedded, total = embedding_counts(session)
            update_embedding_run(
                session,
                run_id,
                embedded_chunk_count=embedded,
                complete=embedded == total,
            )
        newly_embedded += len(batch)

    with session_scope() as session:
        embedded, total = embedding_counts(session)
        if embedded != total:
            raise RuntimeError(f"embedding run incomplete: {embedded}/{total} chunks")
        update_embedding_run(
            session, run_id, embedded_chunk_count=embedded, complete=True
        )
    return {
        "run_id": run_id,
        "model": encoder.model_name,
        "dimensions": encoder.dimensions,
        "normalized": True,
        "newly_embedded_chunks": newly_embedded,
        "embedded_chunks": embedded,
        "total_chunks": total,
        "corpus_arxiv_ids_sha256": manifest.arxiv_ids_sha256,
    }
