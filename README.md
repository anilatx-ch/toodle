# TOODLE - Intelligent Support Ticket Classifier

A clean, production-ready implementation of an intelligent support ticket categorization system demonstrating Full-Stack AI Engineer capabilities.

## Project Goals

TOODLE demonstrates end-to-end ML engineering with clean, maintainable code (~5,200 lines):

- **Traditional ML**: CatBoost + XGBoost
- **Deep Learning**: DistilBERT with KerasNLP
- **Data Engineering**: dbt + DuckDB pipeline
- **MLOps**: MLflow experiment tracking + Optuna tuning
- **RAG Retrieval**: FAISS + entity search
- **Anomaly Detection**: Confidence & volume-based monitoring
- **Production API**: FastAPI serving with Docker deployment

## Key Design Decision

All models train on **clean, deduplicated data** (~110 unique subject→category mappings) rather than noisy 100K samples, achieving >85% F1 score. The 100K corpus is used for RAG retrieval and anomaly baselines.

## Quick Architecture

```
support_tickets.json (100K)
    ├─→ dbt Pipeline → featured_tickets (RAG corpus, search index)
    └─→ Splitter → clean_training (~110) → Model Training
                                              ├─ CatBoost
                                              ├─ XGBoost
                                              └─ DistilBERT
                                                    ↓
                                              FastAPI Serving
                                              ├─ /predict (category)
                                              ├─ /analyze-feedback (sentiment)
                                              ├─ /search (RAG retrieval)
                                              └─ /health
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed component design.

## Quick Start

### Prerequisites

Ensure you have the required data file:
- `support_tickets.json` (must be provided - see project requirements)

### Installation

```bash
# 1. Install system packages (one-time setup, requires sudo)
make install-system

# 2. Install Python, Poetry, and dependencies (verifies system packages)
make install

# 3. Run tests to verify installation
make test
```

**Note:** If you already have system packages installed, `make install` will verify and proceed automatically. Only run `make install-system` if verification fails.

### Troubleshooting

**Error: Missing required system packages**
```bash
ERROR: Missing required system packages: libncurses-dev

To install missing packages, run:
  make install-system
```

**Solution:** Run `make install-system` to install the missing packages, then retry `make install`.

## API Usage

### Start the API

```bash
# Local development (with auto-reload)
make api

# Docker deployment
make docker-build
make docker-up

# API will be available at http://localhost:8000
```

### Example: Predict Ticket Category

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
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
  "warning": "priority_placeholder,sentiment_placeholder",
  "model_used": "xgboost",
  "inference_time_ms": 15.2
}
```

### Example: Analyze Feedback Sentiment

```bash
curl -X POST http://localhost:8000/analyze-feedback \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TK-12345",
    "feedback_text": "Great support! Issue resolved quickly."
  }'
```

**Response:**
```json
{
  "ticket_id": "TK-12345",
  "predicted_sentiment": "satisfied",
  "sentiment_confidence": 0.89,
  "model_used": "catboost_sentiment",
  "inference_time_ms": 8.7
}
```

### Example: Search Similar Resolutions

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Database sync timeout ERROR_TIMEOUT_429",
    "top_k": 5,
    "filters": {"category": "Technical Issue"}
  }'
```

**Response:**
```json
{
  "results": [
    {
      "ticket_id": "TK-9876",
      "resolution": "Increased connection timeout to 60s in sync_engine config.",
      "category": "Technical Issue",
      "similarity_score": 0.94,
      "matched_entities": ["ERROR_TIMEOUT_429", "DataSync Pro"]
    }
  ],
  "search_time_ms": 23.5
}
```

See [docs/API_CONTRACT.md](docs/API_CONTRACT.md) for complete API documentation.

## Documentation

### Core Documentation
- [API Contract](docs/API_CONTRACT.md) - Endpoint specifications and examples
- [Technical Decisions](docs/DECISIONS.md) - Decision log with rationale (D-001 to D-020)
- [System Architecture](docs/ARCHITECTURE.md) - Component design and data flow
- [Model Documentation](docs/MODEL.md) - Performance analysis and comparisons

### Assessment Reference
- [Original Specification](0_OBJECTIVE.md) - Assessment requirements
- [Submission Checklist](docs/SUBMISSION_PACKAGE.md) - Included/excluded files and packaging details

### Investigations
- [Subcategory Independence Analysis](exploration/subcategory_independence/REPORT.md) - Statistical evidence for scope decisions
- [Exploration Methodology](exploration/README.md) - Investigation approach

## Project Status

✅ **All stages complete** (Stage 0-7)

- **Stage 0**: Project scaffold, configuration system
- **Stage 1**: Data pipeline (dbt + DuckDB, clean training set extraction)
- **Stage 2**: Feature engineering (TF-IDF, categorical encoding)
- **Stage 2.5**: Evaluation infrastructure (metrics, MLflow, reporting)
- **Stage 3**: Traditional ML (CatBoost, XGBoost)
- **Stage 4**: Deep learning (DistilBERT)
- **Stage 4.5**: Model comparison and reporting
- **Stage 5**: Sentiment, search, anomaly detection
- **Stage 6**: API & integration
- **Stage 7**: Documentation polish & verification

**Current Metrics:**
- 87 tests passing
- 5,181 source lines of code
- 3 trained classification models (CatBoost, XGBoost, DistilBERT)
- 4 API endpoints
