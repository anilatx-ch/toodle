# TOODLE - Intelligent Support Ticket Classifier

**Status:** Under construction (porting from DOODLE)

A clean, production-ready implementation of an intelligent support ticket categorization system demonstrating Full-Stack AI Engineer capabilities.

## Project Goals

TOODLE is a cleaned port of the DOODLE project, reducing ~10,200 lines to ~4,500 lines while maintaining all core functionality:

- **Traditional ML**: CatBoost + XGBoost
- **Deep Learning**: DistilBERT with KerasNLP
- **Data Engineering**: dbt + DuckDB pipeline
- **MLOps**: MLflow experiment tracking + Optuna tuning
- **RAG Retrieval**: FAISS + entity search
- **Production API**: FastAPI serving with Docker deployment

## Key Design Decision

All models train on **clean, deduplicated data** (~110 unique subject→category mappings) rather than noisy 100K samples, targeting >85% F1 score.

## Current Stage

✅ **Stage 0: Scaffold & Config** - Complete
✅ **Stage 1: Data Pipeline** - Complete
- Dual-output pipeline: full 100K corpus + clean ~110 training set
- Clean subject→category extraction and balancing
- Stratified train/val/test split
- dbt integration
- 19 tests passing

🚧 **Next: Stage 2: Feature Engineering**

## Quick Start

```bash
# Install dependencies
make install

# Run tests
make test
```

## Documentation

- `TASK_PORTING.md` - Porting plan and stages
- `AGENTS_PORTING.md` - Code quality guidelines
- `0_OBJECTIVE.md` - Original assessment specification
