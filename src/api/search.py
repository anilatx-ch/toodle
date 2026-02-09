"""Search API endpoint for RAG-based solution retrieval."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.retrieval.search import SearchEngine

router = APIRouter(prefix="", tags=["search"])

_search_engine: Optional[SearchEngine] = None


def get_search_engine() -> SearchEngine:
    """Get or create search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
    return _search_engine


class SearchFilters(BaseModel):
    """Optional metadata filters for search results."""

    category: Optional[str] = None
    product: Optional[str] = None
    resolution_code: Optional[str | list[str]] = None


class SearchRequest(BaseModel):
    """Request schema for /search endpoint."""

    query: str = Field(description="Ticket query text (subject + description)")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: Optional[SearchFilters] = Field(default=None, description="Optional metadata filters")
    include_entities: bool = Field(default=True, description="Extract and boost by entities")


class SearchResultItem(BaseModel):
    """Individual search result."""

    ticket_id: str
    resolution: str
    resolution_code: str
    category: str
    subcategory: str
    product: str
    similarity_score: float = Field(description="Cosine similarity score (0-1)")
    matched_entities: list[str] = Field(description="Entities from query that matched this result")


class SearchResponse(BaseModel):
    """Response schema for /search endpoint."""

    results: list[SearchResultItem]
    query_entities: list[str] = Field(description="Entities extracted from query")
    total_corpus_size: int = Field(description="Total number of indexed resolutions")
    search_time_ms: float = Field(description="Search latency in milliseconds")


class SearchErrorResponse(BaseModel):
    """Error response when search index is not ready."""

    error: str
    details: str
    action: str


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={503: {"model": SearchErrorResponse}},
    summary="Search historical resolutions",
    description=(
        "Semantic search over past ticket resolutions using DistilBERT embeddings + entity matching. "
        "Returns top-k relevant resolutions with similarity scores."
    ),
)
def search(request: SearchRequest):
    """Search for relevant resolutions from historical tickets."""
    engine = get_search_engine()

    if not engine.is_ready:
        return JSONResponse(
            status_code=503,
            content=SearchErrorResponse(
                error="search_index_not_ready",
                details="Search index has not been built. Run: make build-search-index",
                action="build_search_index",
            ).model_dump(),
        )

    try:
        result = engine.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters.model_dump() if request.filters else None,
            include_entities=request.include_entities,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=SearchErrorResponse(
                error="search_failed",
                details=str(exc),
                action="check_logs",
            ).model_dump(),
        )

    result_items = [SearchResultItem(**item) for item in result.results]
    return SearchResponse(
        results=result_items,
        query_entities=result.query_entities,
        total_corpus_size=result.total_corpus_size,
        search_time_ms=result.search_time_ms,
    )
