"""Entity extraction and inverted-index utilities for retrieval boosting."""

from __future__ import annotations

import json
from pathlib import Path

from src.features.entities import ERROR_RE, PRODUCTS


def extract_entities(text: str) -> set[str]:
    """Extract known error codes and product names from query text."""
    entities = set(ERROR_RE.findall(text))
    lowered = text.lower()
    for product in PRODUCTS:
        if product.lower() in lowered:
            entities.add(product)
    return entities


class EntityIndex:
    """Maps entities to document ids for score boosting."""

    def __init__(self) -> None:
        self.index: dict[str, set[str]] = {}

    def build(self, corpus: list) -> None:
        self.index = {}
        for doc in corpus:
            for code in doc.error_codes:
                self.index.setdefault(code, set()).add(doc.ticket_id)
            if doc.product:
                self.index.setdefault(doc.product, set()).add(doc.ticket_id)
            for tag in doc.tags:
                self.index.setdefault(tag, set()).add(doc.ticket_id)

    def lookup(self, entities: set[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entity in entities:
            for doc_id in self.index.get(entity, set()):
                counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

    def boost_results(
        self,
        results: list[tuple[str, float]],
        query_entities: set[str],
        boost_factor: float = 0.15,
    ) -> list[tuple[str, float]]:
        if not query_entities:
            return results
        matches = self.lookup(query_entities)
        boosted: list[tuple[str, float]] = []
        for doc_id, score in results:
            boosted.append((doc_id, score * (1.0 + boost_factor * matches.get(doc_id, 0))))
        boosted.sort(key=lambda item: item[1], reverse=True)
        return boosted

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {entity: sorted(doc_ids) for entity, doc_ids in self.index.items()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EntityIndex":
        if not path.exists():
            raise FileNotFoundError(f"Entity index not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        index = cls()
        index.index = {entity: set(doc_ids) for entity, doc_ids in data.items()}
        return index

