"""Generate retrieval embeddings from the resolution corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.retrieval.corpus import load_corpus  # noqa: E402
from src.retrieval.embeddings import EmbeddingService  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate retrieval embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_directories()

    corpus = load_corpus(limit=args.limit)
    if not corpus:
        raise RuntimeError("No resolution documents found for embedding generation")

    embedder = EmbeddingService()
    texts = [doc.source_text() for doc in corpus]
    embeddings = embedder.embed_batch(texts, batch_size=args.batch_size)
    ticket_ids = np.asarray([doc.ticket_id for doc in corpus], dtype=object)

    config.EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_PATH, embeddings)
    np.save(config.EMBEDDINGS_TICKET_IDS_PATH, ticket_ids)
    metadata = {
        "rows": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "batch_size": args.batch_size,
        "env": config.ENV,
        "smoke_test": config.SMOKE_TEST,
    }
    config.EMBEDDINGS_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Embeddings: {config.EMBEDDINGS_PATH}")
    print(f"Ticket IDs: {config.EMBEDDINGS_TICKET_IDS_PATH}")
    print(f"Metadata: {config.EMBEDDINGS_METADATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

