# TOODLE - Project Context

## Overview

TOODLE is an intelligent support ticket system demonstrating Full-Stack AI Engineer skills. This is a technical assessment showcasing clean, maintainable code (~4,500 lines) that delivers production-ready ML capabilities while avoiding over-engineering.

**Goal**: Demonstrate clean design, practical trade-offs, and production thinking within a constrained scope (approximately one working day of development).

## Current Stage

**Status**: ✅ Stage 7 (Documentation Polish & Final Verification) complete
**Latest commit**: [To be created after user review]
**Next**: Project complete - ready for assessment submission

### What's Been Built

- **Stage 0**: Project scaffold, configuration system, build automation, initial documentation
- **Stage 1**: Dual-output data pipeline
  - Full corpus (100K tickets) through dbt for RAG/search
  - Clean training set (~110 deduplicated samples) for model training
  - DuckDB staging and transformation
  - Stratified train/val/test splits
- **Stage 2**: Feature Engineering
  - TF-IDF text features with Chi2 selection (10K vocab → 5K features)
  - Categorical one-hot encoding
  - Numerical standard scaling
  - Entity extraction (error codes, products, temporal)
  - Classifier-specific feature pipelines
- **Stage 2.5**: Evaluation & Experiment Tracking Infrastructure
  - Evaluation metrics (F1, accuracy, precision, recall, ECE)
  - Error analysis and confusion cluster generation
  - Latency benchmarking utilities
  - MLflow logging utilities
  - Evaluation orchestration script
- **Stage 3**: Traditional ML Training
  - CatBoost and XGBoost model wrappers
  - Training pipelines on clean train/val/test split parquet
  - Optional Optuna tuning with bounded trial counts for small clean dataset
  - MLflow logging, evaluation summaries, and latency profiling
  - Training orchestration script (`scripts/run_training.py`)
- **Stage 4**: Deep Learning (BERT)
  - DistilBERT wrapper with text-only default and optional tabular branch
  - Clean-data training entrypoint (`src/training/train_bert.py`)
  - Model metadata + weights serialization (`metadata.json` + `model.weights.h5`)
  - Makefile targets: `download-bert`, `train-bert`
- **Stage 5**: Sentiment, Search & Anomaly
  - Sentiment classifier training (`src/training/train_sentiment.py`)
  - Retrieval modules (`src/retrieval/*`) with FAISS + entity index
  - Anomaly modules (`src/anomaly/*`) for confidence and volume analysis
  - Build scripts: `build_search_index.py`, `generate_embeddings.py`, `build_anomaly_baseline.py`
  - Makefile targets: `train-sentiment`, `build-search-index`, `build-anomaly-baseline`
- **Stage 6**: API & Integration
  - FastAPI application (`src/api/app.py`) with /predict, /analyze-feedback, /search, /health
  - Pydantic schemas (`src/api/schemas.py`) with placeholder field support
  - Search router (`src/api/search.py`)
  - ModelManager for loading CatBoost + XGBoost + BERT + sentiment models
  - API tests: 9 tests covering all endpoints
  - Makefile targets: `api`, `docker-build`, `docker-up`, `docker-down`
  - Documentation: API_CONTRACT.md, D-019, D-020

### Current Metrics

- **Source code**: 5,181 LOC in `src/` (Stage 7 complete)
- **Tests**: 87 passing
- **Documentation**: 5 core docs (README, ARCHITECTURE, DECISIONS, MODEL, API_CONTRACT)
- **Data quality**: Zero label conflicts in clean training set (was 30% in noisy 100K)
- **Feature dimensions**: ~5056 total (5000 TF-IDF + ~50 categorical + ~6 numerical)
- **API endpoints**: 4 (/predict, /analyze-feedback, /search, /health)

## Scope Boundaries

### Included (demonstrates assessment requirements)

**Traditional ML**: CatBoost + XGBoost (2 backends for comparison)
**Deep Learning**: DistilBERT fine-tuning via TensorFlow/Keras
**Data Engineering**: dbt + DuckDB pipeline
**Experiment Tracking**: MLflow local tracking
**RAG/Retrieval**: FAISS vector search + entity keyword matching
**Monitoring**: Volume-based anomaly detection
**API**: FastAPI serving with /predict, /search, /analyze-feedback endpoints
**DevOps**: Docker + docker-compose, Makefile automation

### Out of Scope

- LightGBM (3rd traditional ML backend - not needed for comparison)
- Subcategory classification (deferred - see investigation report)
- Training on noisy 100K dataset (data quality issue - use clean deduplicated set)
- Experimental analysis scripts
- 3-tier environment config (simplified to SMOKE_TEST + ENV)

### Placeholder Features

- **Priority prediction**: Returns "medium" deterministically in /predict (shows multi-output design intent)
- **Sentiment in /predict**: Returns "neutral" deterministically (real analysis via /analyze-feedback)

See [AGENTS_PORTING.md](AGENTS_PORTING.md) for detailed scope and quality guidelines.

## Technology Stack

**Language**: Python 3.12
**ML Frameworks**: TensorFlow/Keras (BERT), CatBoost, XGBoost, scikit-learn
**Data Pipeline**: dbt, DuckDB (in-process)
**Experiment Tracking**: MLflow (local file store)
**Feature Engineering**: TF-IDF, one-hot encoding, standard scaling
**Retrieval**: FAISS (vector search), spaCy (entity extraction)
**API**: FastAPI, Pydantic
**Testing**: pytest
**Build**: Make, Poetry
**Deployment**: Docker, docker-compose

## Key Architecture Decisions

These are the critical design choices that define the system. See [docs/DECISIONS.md](docs/DECISIONS.md) for full rationale.

**D-006: Clean Training Data Strategy**
- Train all models on ~110 deduplicated subject→category pairs (not noisy 100K)
- Rationale: Subject→category mapping is deterministic. The 100K dataset has 30% label noise from generation process.
- Expected: >85% F1 vs 18% on noisy data

**D-007: Dual Output Pipeline**
- Full 100K through dbt → RAG corpus, anomaly baselines
- Clean ~110 through splitter → model training
- Two data paths, distinct purposes

**D-001: Traditional ML Frameworks**
- CatBoost + XGBoost (drop LightGBM)
- Two gradient boosting implementations suffice for comparison

**D-002: Deep Learning Framework**
- DistilBERT via keras-nlp on TensorFlow
- Assessment specifies "TensorFlow/Keras"

**D-003: Data Pipeline Stack**
- dbt + DuckDB (in-process)
- Demonstrates data engineering without infrastructure overhead

## Data Quality Finding

**The Critical Insight**: Subject→category mapping is deterministic with ~110 unique pairs.

**Evidence**:
- 100K noisy dataset: 30% conflicting labels → 18% F1 with CatBoost
- ~110 clean templates: Zero conflicts → 88% F1 with CatBoost

**Decision**: Train on clean templates, use full 100K for RAG corpus only.

This finding fundamentally changed the training approach and is documented in [docs/MODEL.md](docs/MODEL.md#data-quality-investigation).

## Current Validation Status

**Tests**: 87 passing (pytest)
- Includes Stage 3 model wrapper and training orchestrator coverage
- Includes Stage 4 additions (`tests/test_bert_model.py`: 5 passing)
- Includes Stage 4.5 additions (`tests/test_generate_report.py`: 2 passing)
- Includes Stage 5 additions (`tests/test_search.py`: 5 passing, `tests/test_anomaly.py`: 9 passing)
- Includes Stage 6 additions (`tests/test_api.py`: 9 passing)

**Smoke mode**: `SMOKE_TEST=true` uses 100-record subset for fast validation

**Known issues**: None currently

## Quick Navigation

**Porting and Planning**:
- [PLAN_PORTING.md](PLAN_PORTING.md) - Full 7-stage porting roadmap
- [AGENTS_PORTING.md](AGENTS_PORTING.md) - Quality guidelines and scope boundaries
- [TASK_PORTING.md](TASK_PORTING.md) - Task tracking

**Technical Documentation**:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and data flow (living doc)
- [docs/DECISIONS.md](docs/DECISIONS.md) - Technical decision log D-001 to D-020 (living doc)
- [docs/MODEL.md](docs/MODEL.md) - Model performance and data quality analysis (living doc)
- [docs/API_CONTRACT.md](docs/API_CONTRACT.md) - API endpoint specifications and examples

**Setup and Usage**:
- [README.md](README.md) - Quick start, installation, basic usage
- [0_OBJECTIVE.md](0_OBJECTIVE.md) - Original assessment specification

## Stage Progress Tracker

- ✅ **Stage 0: Scaffold & Config**
  - pyproject.toml, Makefile, config.py, Docker setup
  - AGENTS_PORTING.md, initial README
  - Tests: config import, ENV handling

- ✅ **Stage 1: Data Pipeline**
  - loader.py, splitter.py, schema.py, hierarchy.py
  - dbt project with featured_tickets model
  - Dual output: full corpus + clean training set
  - Tests: 19 passing (data pipeline, hierarchy)
  - Documentation: DECISIONS.md (D-006, D-007), ARCHITECTURE.md (data flow)
  - **Current**: 446 LOC in src/

- ✅ **Stage 2: Feature Engineering** (complete)
  - TF-IDF vectorization (10K vocab → 5K Chi2 selected)
  - Categorical/ordinal encoding, numerical scaling
  - Entity extraction (error codes, products, temporal)
  - Classifier-specific pipelines (category, priority, sentiment)
  - Tests: 42 passing (23 new tests)
  - Documentation: DECISIONS.md (D-008, D-009), ARCHITECTURE.md (Feature Engineering section)
  - **Current**: ~850 LOC in src/

- ✅ **Stage 2.5: Evaluation & Experiment Tracking** (complete)
  - Evaluation metrics: F1, accuracy, precision, recall, ECE
  - Error analysis: confusion clusters, confidence analysis
  - Latency benchmarking utilities
  - MLflow logging utilities with fallback handling
  - Evaluation orchestration (run_evaluation.py)
  - Tests: 48 passing (6 new tests)
  - Documentation: DECISIONS.md (D-010, D-011), docs/MODEL.md (initial structure)
  - **Current**: 2104 LOC in src/

- ✅ **Stage 3: Traditional ML Models** (complete)
  - CatBoost and XGBoost training on clean ~110 samples
  - Optuna hyperparameter tuning
  - MLflow experiment tracking
  - Evaluation metrics and latency analysis
  - Training orchestration via `scripts/run_training.py`
  - Tests: 57 passing (9 new tests)
  - Documentation: DECISIONS.md (D-013), ARCHITECTURE.md (Stage 3 training), MODEL.md (traditional ML section)
  - **Current**: 2952 LOC in src/

- ✅ **Stage 4: Deep Learning (BERT)** (complete)
  - DistilBERT wrapper implemented in `src/models/bert_model.py`
  - Text-only training entrypoint implemented in `src/training/train_bert.py`
  - Makefile targets added: `download-bert`, `train-bert`
  - Tests: `tests/test_bert_model.py` (5 passing)
  - Documentation: DECISIONS.md (D-014, D-015), MODEL.md (BERT section)

- ✅ **Stage 4.5: Model Comparison & Reporting** (complete)
  - Report generation script (`scripts/generate_report.py`)
  - Automated MODEL.md updates with performance comparison table
  - Standalone comparison report generation
  - Model recommendation logic based on F1 and latency
  - Tests: `tests/test_generate_report.py` (2 passing)
  - Makefile target: `report`
  - Documentation: automatically updates docs/MODEL.md comparison section

- ✅ **Stage 5: Sentiment, Search & Anomaly** (complete)
  - Sentiment CatBoost training pipeline implemented (`src/training/train_sentiment.py`)
  - Retrieval stack implemented (`src/retrieval/corpus.py`, `src/retrieval/embeddings.py`, `src/retrieval/index.py`, `src/retrieval/entities.py`, `src/retrieval/search.py`)
  - Anomaly stack implemented (`src/anomaly/detector.py`, `src/anomaly/baselines.py`, `src/anomaly/volume_analyzer.py`)
  - Build scripts added: `scripts/build_search_index.py`, `scripts/generate_embeddings.py`, `scripts/build_anomaly_baseline.py`
  - Tests added: `tests/test_search.py` (5 passing), `tests/test_anomaly.py` (9 passing)
  - Makefile targets added: `train-sentiment`, `build-search-index`, `build-anomaly-baseline`

- ✅ **Stage 6: API & Integration** (complete)
  - FastAPI application: `src/api/app.py` (421 LOC)
  - Endpoints: /predict (category + placeholders), /analyze-feedback (sentiment), /search (RAG), /health
  - Schemas: `src/api/schemas.py` (99 LOC) with placeholder field support
  - Search router: `src/api/search.py` (120 LOC)
  - ModelManager: loads 2 trad ML + BERT + sentiment + anomaly detector
  - Tests: `tests/test_api.py` (9 passing)
  - Makefile targets: `api`, `docker-build`, `docker-up`, `docker-down`
  - Documentation: `docs/API_CONTRACT.md`, D-019 (placeholder fields), D-020 (single backend)
  - **Current**: 5137 LOC in `src/`

- ✅ **Stage 7: Documentation Polish & Final Verification** (complete)
  - README.md: Removed "under construction", added API examples, updated metrics
  - ARCHITECTURE.md: Added comprehensive system overview diagram
  - MODEL.md: Complete rewrite with smoke test vs expected performance, training instructions
  - DECISIONS.md: All 20 decisions documented (D-001 to D-020)
  - API_CONTRACT.md: Complete with all 4 endpoint specifications
  - CONTEXT.md: Updated to reflect Stage 7 completion
  - Verification: 87 tests passing, 5,181 LOC, all docs cross-referenced
  - **Current**: 5,181 LOC in src/

---

**Last updated**: Stage 7 complete (Feb 9, 2026)
