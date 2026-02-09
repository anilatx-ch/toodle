# TOODLE - Project Context

## Overview

TOODLE is a cleaned and focused rewrite of the DOODLE intelligent support ticket system. This is a technical assessment demonstrating Full-Stack AI Engineer skills, porting a bloated ~10,200-line codebase down to a maintainable ~4,500-line system that preserves all key capabilities while eliminating redundancy and over-engineering.

**Goal**: Demonstrate clean design, practical trade-offs, and production thinking within a constrained scope (approximately one working day of development).

## Current Stage

**Status**: ✅ Stage 1 (Data Pipeline) complete
**Latest commit**: `5b2b47a - Complete phase 1` (Feb 9, 2026)
**Next**: Stage 2 (Feature Engineering)

### What's Been Built

- **Stage 0**: Project scaffold, configuration system, build automation, initial documentation
- **Stage 1**: Dual-output data pipeline
  - Full corpus (100K tickets) through dbt for RAG/search
  - Clean training set (~110 deduplicated samples) for model training
  - DuckDB staging and transformation
  - Stratified train/val/test splits

### Current Metrics

- **Source code**: 446 lines in `src/`
- **Tests**: 19 passing
- **Data quality**: Zero label conflicts in clean training set (was 30% in noisy 100K)

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

### Excluded (trimmed from DOODLE)

- LightGBM (3rd traditional ML backend - redundant)
- Subcategory classification (5 duplicate feature configs - deferred)
- Training on noisy 100K dataset (data quality bug, not feature)
- Experimental scripts (deterministic_mappings.py, single_feature_power.py, etc.)
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

**Tests**: 19 passing (pytest)
- Configuration import and path resolution
- Data pipeline (loader, splitter, hierarchy)
- dbt model tests

**Smoke mode**: `SMOKE_TEST=true` uses 100-record subset for fast validation

**Known issues**: None currently

## Quick Navigation

**Porting and Planning**:
- [PLAN_PORTING.md](PLAN_PORTING.md) - Full 7-stage porting roadmap
- [AGENTS_PORTING.md](AGENTS_PORTING.md) - Quality guidelines and scope boundaries
- [TASK_PORTING.md](TASK_PORTING.md) - Task tracking

**Technical Documentation**:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and data flow (living doc)
- [docs/DECISIONS.md](docs/DECISIONS.md) - Technical decision log D-001 to D-007 (living doc)
- [docs/MODEL.md](docs/MODEL.md) - Model performance and data quality analysis (living doc)

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

- 🚧 **Stage 2: Feature Engineering** (next)
  - TF-IDF text features
  - Categorical and numerical feature encoding
  - Feature pipeline with train/val/test transforms
  - Expected tests: ~24 passing

- ⏳ **Stage 3: Traditional ML Models**
  - CatBoost and XGBoost training on clean ~110 samples
  - Optuna hyperparameter tuning
  - MLflow experiment tracking
  - Evaluation metrics and latency analysis
  - Expected: >85% F1 on category prediction

- ⏳ **Stage 4: Deep Learning (BERT)**
  - DistilBERT fine-tuning on clean ~110 samples
  - Text-only training path (multimodal as option)
  - Model comparison report (BERT vs CatBoost vs XGBoost)
  - Expected: >85% F1, comparison with traditional ML

- ⏳ **Stage 5: Sentiment, Search & Anomaly**
  - Sentiment classifier (CatBoost on feedback text)
  - FAISS vector search + entity keyword matching
  - Volume-based anomaly detection
  - Build indices and baselines

- ⏳ **Stage 6: API & Integration**
  - FastAPI endpoints: /predict, /search, /analyze-feedback, /health
  - ModelManager loading CatBoost + XGBoost + BERT
  - Confidence-weighted ensemble predictions
  - Full smoke test pipeline (data → train → serve)

- ⏳ **Stage 7: Documentation Polish**
  - Fill all performance numbers in docs/MODEL.md
  - Complete API examples in README.md
  - System diagram in docs/ARCHITECTURE.md
  - Final verification: tests, pipeline, Docker, docs

---

**Last updated**: Stage 1 complete (Feb 9, 2026)
