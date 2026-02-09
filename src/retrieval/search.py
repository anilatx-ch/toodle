"""Search orchestration for semantic + entity-aware retrieval."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src import config
from src.retrieval.corpus import ResolutionDocument, load_corpus
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.entities import EntityIndex, extract_entities
from src.retrieval.index import FAISSIndex


@dataclass(frozen=True)
class SearchResult:
    """Search response payload."""

    results: list[dict[str, object]]
    query_entities: list[str]
    total_corpus_size: int
    search_time_ms: float


class SearchEngine:
    """Semantic search with optional entity-based score boosting."""

    def __init__(self, embedder: EmbeddingService | None = None) -> None:
        self._faiss_index: FAISSIndex | None = None
        self._entity_index: EntityIndex | None = None
        self._corpus: dict[str, ResolutionDocument] = {}
        self._embedder = embedder or EmbeddingService()

    @property
    def is_ready(self) -> bool:
        return (
            config.FAISS_INDEX_PATH.exists()
            and config.RETRIEVAL_ENTITY_INDEX_PATH.exists()
            and config.RETRIEVAL_CORPUS_PATH.exists()
        )

    def _ensure_loaded(self) -> None:
        if self._faiss_index is not None:
            return
        self._faiss_index = FAISSIndex.load(config.FAISS_INDEX_PATH)
        self._entity_index = EntityIndex.load(config.RETRIEVAL_ENTITY_INDEX_PATH)
        documents = load_corpus(config.RETRIEVAL_CORPUS_PATH, limit=None)
        self._corpus = {doc.ticket_id: doc for doc in documents}

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        include_entities: bool = True,
        boost_factor: float = 0.15,
    ) -> SearchResult:
        if not query.strip():
            raise ValueError("Query must not be empty")

        start = time.perf_counter()
        self._ensure_loaded()

        query_entities = extract_entities(query) if include_entities else set()
        query_vector = self._embedder.embed_query(query)
        candidate_k = max(top_k, top_k * 3 if filters else top_k)
        candidates = self._faiss_index.search(query_vector, k=candidate_k)
        if include_entities and query_entities:
            candidates = self._entity_index.boost_results(
                candidates,
                query_entities,
                boost_factor=boost_factor,
            )
        if filters:
            candidates = self._apply_filters(candidates, filters)

        payload: list[dict[str, object]] = []
        for doc_id, score in candidates[:top_k]:
            doc = self._corpus.get(doc_id)
            if doc is None:
                continue
            matched_entities = [
                entity
                for entity in query_entities
                if entity in doc.error_codes or entity == doc.product
            ]
            payload.append(
                {
                    "ticket_id": doc.ticket_id,
                    "resolution": doc.resolution_text,
                    "resolution_code": doc.resolution_code,
                    "category": doc.category,
                    "product": doc.product,
                    "similarity_score": float(score),
                    "matched_entities": matched_entities,
                    "related_tickets": doc.related_tickets,
                    "kb_articles_helpful": doc.kb_articles_helpful,
                    "kb_articles_viewed": doc.kb_articles_viewed,
                    "auto_suggested_solutions": doc.auto_suggested_solutions,
                    "auto_suggestion_accepted": doc.auto_suggestion_accepted,
                    "resolution_helpful": doc.resolution_helpful,
                    "agent_actions": doc.agent_actions,
                    "resolution_time_hours": doc.resolution_time_hours,
                    "satisfaction_score": doc.satisfaction_score,
                    "template_used": doc.template_used,
                    "tags": doc.tags,
                }
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SearchResult(
            results=payload,
            query_entities=sorted(query_entities),
            total_corpus_size=len(self._corpus),
            search_time_ms=elapsed_ms,
        )

    def _apply_filters(
        self,
        candidates: list[tuple[str, float]],
        filters: dict[str, object],
    ) -> list[tuple[str, float]]:
        filtered: list[tuple[str, float]] = []
        for doc_id, score in candidates:
            doc = self._corpus.get(doc_id)
            if doc is None:
                continue
            if "category" in filters and doc.category != filters["category"]:
                continue
            if "product" in filters and doc.product != filters["product"]:
                continue
            resolution_code_filter = filters.get("resolution_code")
            if isinstance(resolution_code_filter, list):
                if doc.resolution_code not in resolution_code_filter:
                    continue
            elif resolution_code_filter is not None and doc.resolution_code != resolution_code_filter:
                continue
            filtered.append((doc_id, score))
        return filtered

