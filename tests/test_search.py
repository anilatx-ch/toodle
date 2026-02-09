"""Unit tests for retrieval modules."""

# ruff: noqa: E402

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("faiss")

from src.retrieval.corpus import ResolutionDocument, load_corpus
from src.retrieval.entities import EntityIndex, extract_entities
from src.retrieval.index import FAISSIndex
from src.retrieval.search import SearchEngine


def test_extract_entities() -> None:
    entities = extract_entities("sync fails with ERROR_TIMEOUT_429 in DataSync Pro")
    assert "ERROR_TIMEOUT_429" in entities
    assert "DataSync Pro" in entities


def test_load_corpus_filters_empty_resolutions(tmp_path) -> None:
    payload = [
        {
            "ticket_id": "TK-1",
            "resolution": "Restart service",
            "description": "ERROR_SERVICE_500",
            "product": "API Gateway",
        },
        {"ticket_id": "TK-2", "resolution": "", "description": "should skip"},
    ]
    path = tmp_path / "tickets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    docs = load_corpus(path, limit=None)
    assert len(docs) == 1
    assert docs[0].ticket_id == "TK-1"
    assert "ERROR_SERVICE_500" in docs[0].error_codes


def test_entity_index_lookup_and_boost() -> None:
    docs = [
        ResolutionDocument(
            ticket_id="A",
            resolution_text="Fix timeout",
            resolution_code="FIX",
            category="Technical Issue",
            product="DataSync Pro",
            subject="timeout",
            description="ERROR_TIMEOUT_429",
            error_codes=["ERROR_TIMEOUT_429"],
            template_used=None,
            related_tickets=[],
            kb_articles_helpful=[],
            kb_articles_viewed=[],
            auto_suggested_solutions=[],
            auto_suggestion_accepted=False,
            resolution_helpful=None,
            agent_actions=[],
            resolution_time_hours=None,
            satisfaction_score=None,
            tags=[],
        ),
        ResolutionDocument(
            ticket_id="B",
            resolution_text="Rotate key",
            resolution_code="SEC",
            category="Security",
            product="API Gateway",
            subject="auth",
            description="ERROR_AUTH_401",
            error_codes=["ERROR_AUTH_401"],
            template_used=None,
            related_tickets=[],
            kb_articles_helpful=[],
            kb_articles_viewed=[],
            auto_suggested_solutions=[],
            auto_suggestion_accepted=False,
            resolution_helpful=None,
            agent_actions=[],
            resolution_time_hours=None,
            satisfaction_score=None,
            tags=[],
        ),
    ]
    entity_index = EntityIndex()
    entity_index.build(docs)
    counts = entity_index.lookup({"ERROR_TIMEOUT_429", "DataSync Pro"})
    assert counts == {"A": 2}

    boosted = entity_index.boost_results([("A", 0.8), ("B", 0.79)], {"DataSync Pro"})
    assert boosted[0][0] == "A"
    assert boosted[0][1] > boosted[1][1]


def test_faiss_index_roundtrip(tmp_path) -> None:
    vectors = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = ["doc1", "doc2", "doc3"]

    index = FAISSIndex(dimension=2)
    index.build(vectors, ids)
    assert index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), k=1)[0][0] == "doc1"

    path = tmp_path / "index.bin"
    index.save(path)
    loaded = FAISSIndex.load(path)
    results = loaded.search(np.asarray([[1.0, 0.0]], dtype=np.float32), k=2)
    assert len(results) == 2
    assert results[0][0] == "doc1"


class _StubEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        if "timeout" in text:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)
        return np.asarray([[0.0, 1.0]], dtype=np.float32)


def test_search_engine_filters_and_entities() -> None:
    docs = [
        ResolutionDocument(
            ticket_id="A",
            resolution_text="Tune timeout thresholds",
            resolution_code="CONFIG_CHANGE",
            category="Technical Issue",
            product="DataSync Pro",
            subject="sync timeout",
            description="ERROR_TIMEOUT_429 when syncing",
            error_codes=["ERROR_TIMEOUT_429"],
            template_used=None,
            related_tickets=[],
            kb_articles_helpful=[],
            kb_articles_viewed=[],
            auto_suggested_solutions=[],
            auto_suggestion_accepted=False,
            resolution_helpful=None,
            agent_actions=[],
            resolution_time_hours=None,
            satisfaction_score=None,
            tags=[],
        ),
        ResolutionDocument(
            ticket_id="B",
            resolution_text="Reset API credentials",
            resolution_code="SECURITY_PATCH",
            category="Security",
            product="API Gateway",
            subject="auth issue",
            description="ERROR_AUTH_401 on login",
            error_codes=["ERROR_AUTH_401"],
            template_used=None,
            related_tickets=[],
            kb_articles_helpful=[],
            kb_articles_viewed=[],
            auto_suggested_solutions=[],
            auto_suggestion_accepted=False,
            resolution_helpful=None,
            agent_actions=[],
            resolution_time_hours=None,
            satisfaction_score=None,
            tags=[],
        ),
    ]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = FAISSIndex(dimension=2)
    index.build(vectors, ["A", "B"])

    entity_index = EntityIndex()
    entity_index.build(docs)

    engine = SearchEngine(embedder=_StubEmbedder())  # type: ignore[arg-type]
    engine._faiss_index = index
    engine._entity_index = entity_index
    engine._corpus = {doc.ticket_id: doc for doc in docs}

    result = engine.search(
        query="timeout issue ERROR_TIMEOUT_429 in DataSync Pro",
        top_k=2,
        filters={"category": "Technical Issue"},
    )
    assert result.total_corpus_size == 2
    assert len(result.results) == 1
    assert result.results[0]["ticket_id"] == "A"
    assert "ERROR_TIMEOUT_429" in result.query_entities
