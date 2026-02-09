"""Build semantic + entity retrieval artifacts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.retrieval.corpus import load_corpus, save_corpus  # noqa: E402
from src.retrieval.embeddings import EmbeddingService  # noqa: E402
from src.retrieval.entities import EntityIndex  # noqa: E402
from src.retrieval.index import FAISSIndex  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval index artifacts")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Absolute limit on number of documents")
    parser.add_argument("--sample-rate", type=float, default=None,
                        help="Fraction of documents to use (0.0-1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Measure time and estimate prod runtime without saving artifacts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_directories()

    corpus = load_corpus(limit=args.limit, sample_rate=args.sample_rate)
    if not corpus:
        raise RuntimeError("No resolution documents found")

    embedder = EmbeddingService()

    start_time = time.time()
    vectors = embedder.embed_batch([doc.source_text() for doc in corpus], batch_size=args.batch_size)
    elapsed = time.time() - start_time

    doc_ids = [doc.ticket_id for doc in corpus]

    # Dry-run mode: estimate and exit
    if args.dry_run:
        # Count total eligible documents
        import json
        from src.data.loader import find_raw_json
        raw_path = find_raw_json()
        with raw_path.open(encoding="utf-8") as f:
            raw_rows = json.load(f)
        # Count eligible docs (same filter as load_corpus)
        total_docs = 0
        for row in raw_rows:
            if isinstance(row, dict):
                ticket_id = str(row.get("ticket_id") or "")
                resolution = str(row.get("resolution") or row.get("resolution_text") or "").strip()
                if ticket_id and resolution:
                    total_docs += 1

        sampled_docs = len(corpus)
        estimated_seconds = (elapsed / sampled_docs) * total_docs if sampled_docs > 0 else 0
        estimated_mins = estimated_seconds / 60
        estimated_hours = estimated_mins / 60

        print()
        print("=" * 60)
        print("DRY RUN RESULTS - Prod Build Estimate")
        print("=" * 60)
        print(f"Sample docs processed:  {sampled_docs:,}")
        print(f"Total corpus size:      {total_docs:,}")
        print(f"Sample runtime:         {elapsed:.1f}s")
        print(f"Estimated prod runtime: {estimated_seconds:.0f}s (~{estimated_mins:.0f} min / {estimated_hours:.1f} hours)")
        print("=" * 60)
        return 0

    faiss_index = FAISSIndex(dimension=vectors.shape[1])
    faiss_index.build(vectors, doc_ids)
    faiss_index.save(config.FAISS_INDEX_PATH)

    entity_index = EntityIndex()
    entity_index.build(corpus)
    entity_index.save(config.RETRIEVAL_ENTITY_INDEX_PATH)

    save_corpus(corpus, config.RETRIEVAL_CORPUS_PATH)

    print(f"FAISS index: {config.FAISS_INDEX_PATH}")
    print(f"Entity index: {config.RETRIEVAL_ENTITY_INDEX_PATH}")
    print(f"Corpus: {config.RETRIEVAL_CORPUS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

