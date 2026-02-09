# API Contract

**Version:** 1.0.0
**Status:** Stage 6 Complete
**Last Updated:** 2026-02-09

## Overview

TOODLE provides a FastAPI-based REST API for support ticket classification, sentiment analysis, and solution retrieval.

---

## Endpoints

### POST /predict

Predict category for incoming support tickets. Returns category (real model), priority and sentiment (placeholders).

**Request:**

```json
{
  "ticket_id": "TK-12345",
  "subject": "Database sync failing with timeout",
  "description": "Getting ERROR_TIMEOUT_429 when trying to sync large datasets...",
  "error_logs": "ERROR_TIMEOUT_429: Connection timeout after 30s",
  "stack_trace": null,
  "product": "DataSync Pro",
  "product_module": "sync_engine",
  "channel": "email",
  "customer_tier": "enterprise",
  "environment": "production",
  "language": "en",
  "region": "NA",
  "severity": "P2",
  "account_age_days": 365,
  "account_monthly_value": 5000.0,
  "previous_tickets": 3,
  "product_version_age_days": 45,
  "attachments_count": 2,
  "created_at": "2026-02-09T14:30:00Z"
}
```

**Response:**

```json
{
  "ticket_id": "TK-12345",
  "predicted_category": "Technical Issue",
  "predicted_priority": "medium",
  "predicted_sentiment": "neutral",
  "category_confidence": 0.92,
  "priority_confidence": null,
  "sentiment_confidence": null,
  "category_probabilities": {
    "Technical Issue": 0.92,
    "Account Management": 0.03,
    "Feature Request": 0.02,
    "Billing": 0.02,
    "General Inquiry": 0.01
  },
  "priority_probabilities": {
    "low": 0.0,
    "medium": 1.0,
    "high": 0.0,
    "critical": 0.0
  },
  "sentiment_probabilities": {
    "angry": 0.0,
    "confused": 0.0,
    "frustrated": 0.0,
    "grateful": 0.0,
    "neutral": 1.0,
    "satisfied": 0.0
  },
  "warning": "priority_placeholder,sentiment_placeholder",
  "model_used": "xgboost",
  "inference_time_ms": 15.2
}
```

**Status Codes:**
- `200`: Success
- `422`: Invalid request (missing required fields or preprocessing failed)
- `503`: Model backend not ready

**Notes:**
- `priority` and `sentiment` are **placeholders** (deterministic defaults)
- `priority_confidence` and `sentiment_confidence` are `null` for placeholders
- Real category prediction uses configured backend (CatBoost, XGBoost, or BERT)
- Warning flags indicate: `priority_placeholder`, `sentiment_placeholder`, `low_confidence`, `confidence_anomaly`, `possible_leakage_pattern`

---

### POST /analyze-feedback

Analyze sentiment of customer feedback text after ticket resolution. This is the **real sentiment classifier**.

**Request:**

```json
{
  "ticket_id": "TK-12345",
  "feedback_text": "Great support! The issue was resolved quickly and the explanation was clear."
}
```

**Response:**

```json
{
  "ticket_id": "TK-12345",
  "predicted_sentiment": "satisfied",
  "sentiment_confidence": 0.89,
  "sentiment_probabilities": {
    "angry": 0.01,
    "confused": 0.02,
    "frustrated": 0.03,
    "grateful": 0.15,
    "neutral": 0.05,
    "satisfied": 0.89
  },
  "model_used": "catboost_sentiment",
  "inference_time_ms": 8.7,
  "warning": null
}
```

**Status Codes:**
- `200`: Success
- `422`: Invalid request or analysis failed
- `503`: Sentiment model not ready

**Notes:**
- Empty `feedback_text` returns neutral fallback with 0.0 confidence
- Uses CatBoost model trained on customer feedback

---

### POST /search

Search for similar historical ticket resolutions using semantic search + entity matching.

**Request:**

```json
{
  "query": "Database sync timeout ERROR_TIMEOUT_429",
  "top_k": 10,
  "filters": {
    "category": "Technical Issue",
    "product": "DataSync Pro",
    "resolution_code": null
  },
  "include_entities": true
}
```

**Response:**

```json
{
  "results": [
    {
      "ticket_id": "TK-9876",
      "resolution": "Increased connection timeout to 60s in sync_engine config. Added retry logic for transient failures.",
      "resolution_code": "CONFIG_TIMEOUT_ADJUSTED",
      "category": "Technical Issue",
      "subcategory": "Performance",
      "product": "DataSync Pro",
      "similarity_score": 0.94,
      "matched_entities": ["ERROR_TIMEOUT_429", "DataSync Pro"]
    },
    {
      "ticket_id": "TK-8765",
      "resolution": "Root cause: network latency spike. Configured connection pooling to reduce overhead.",
      "resolution_code": "NETWORK_OPTIMIZATION",
      "category": "Technical Issue",
      "subcategory": "Performance",
      "product": "DataSync Pro",
      "similarity_score": 0.87,
      "matched_entities": ["DataSync Pro"]
    }
  ],
  "query_entities": ["ERROR_TIMEOUT_429", "DataSync Pro"],
  "total_corpus_size": 42156,
  "search_time_ms": 23.5
}
```

**Status Codes:**
- `200`: Success
- `500`: Search failed (unexpected error)
- `503`: Search index not built

**Notes:**
- Requires search index built via `make build-search-index`
- Combines DistilBERT semantic embeddings with entity-based boosting
- Optional filters: `category`, `product`, `resolution_code`

---

### GET /health

System health check with component readiness status.

**Response:**

```json
{
  "status": "ok",
  "contract_version": "1.0.0",
  "serving_backend": "xgboost",
  "backend_ready": true,
  "missing_components": [],
  "anomaly_detector_ready": true,
  "search_index_ready": true
}
```

**Status Code:**
- `200`: Always returns 200 (check `backend_ready` for model status)

**Notes:**
- `backend_ready=false` indicates model artifacts missing
- `missing_components` lists unavailable resources

---

## Error Response Schema

All error responses follow this format:

```json
{
  "ticket_id": "TK-12345",
  "error": "model_not_available",
  "fallback": "manual_triage",
  "details": "xgboost backend not ready: xgboost_model:category"
}
```

**Common error codes:**
- `model_not_available`: Backend not loaded
- `preprocessing_failed`: Feature extraction error
- `sentiment_analysis_failed`: Sentiment classifier error
- `search_index_not_ready`: Search artifacts not built
- `search_failed`: Search execution error

---

## Configuration

**Backend Selection:**
Set via `SERVING_BACKEND` environment variable:
- `catboost` (default for this project)
- `xgboost`
- `bert`

**Environment:**
- `ENV=dev`: Development mode (small models, CUDA disabled)
- `ENV=test`: Test mode (1% data sample)
- `ENV=prod`: Production mode (full dataset)

**Starting the API:**

```bash
# Local development
make api

# Docker
make docker-build
make docker-up
```

---

## Future Integration

**Planned enhancements (not yet implemented):**
- Real priority prediction model
- Real sentiment in /predict response (currently only via /analyze-feedback)
- Pass `predicted_category` from /predict to /search filters for scoped retrieval
- Batch prediction endpoint

---

## Contract Version History

- **1.0.0** (2026-02-09): Initial release with category prediction, sentiment analysis, search, and health endpoints
