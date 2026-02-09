"""FAISS vector index wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:  # pragma: no cover - runtime dependency
    import faiss
except Exception:  # pragma: no cover - runtime dependency
    faiss = None


class FAISSIndex:
    """Exact cosine similarity search with normalized vectors."""

    def __init__(self, dimension: int = 768) -> None:
        if faiss is None:
            raise RuntimeError("FAISS is not installed")
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.id_map: list[str] = []

    def build(self, embeddings: np.ndarray, doc_ids: list[str]) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape={embeddings.shape}")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}"
            )
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("Document id count does not match embedding row count")

        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("Embeddings must be L2 normalized for cosine similarity search")

        self.index.add(embeddings.astype(np.float32))
        self.id_map = list(doc_ids)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if query_embedding.shape != (1, self.dimension):
            raise ValueError(f"Expected query shape (1, {self.dimension}), got {query_embedding.shape}")
        if not self.id_map:
            return []

        k = min(k, len(self.id_map))
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        return [
            (self.id_map[int(idx)], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx != -1
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        sidecar = {"id_map": self.id_map, "dimension": self.dimension}
        path.with_suffix(".id_map.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FAISSIndex":
        if faiss is None:
            raise RuntimeError("FAISS is not installed")
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")

        sidecar_path = path.with_suffix(".id_map.json")
        if not sidecar_path.exists():
            raise FileNotFoundError(f"FAISS id map not found: {sidecar_path}")

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        instance = cls(dimension=int(sidecar["dimension"]))
        instance.index = faiss.read_index(str(path))
        instance.id_map = [str(doc_id) for doc_id in sidecar["id_map"]]
        return instance

