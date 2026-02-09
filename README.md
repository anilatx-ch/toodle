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
- See [D-001 to D-005](docs/DECISIONS.md) for configuration decisions

✅ **Stage 1: Data Pipeline** - Complete
- See [Architecture: Data Pipeline](docs/ARCHITECTURE.md) for design
- See [Model: Pre-Training Baseline](docs/MODEL.md) for expectations
- See [D-006, D-007](docs/DECISIONS.md) for clean data strategy
- 19 tests passing

🚧 **Next: Stage 2: Feature Engineering**

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
ERROR: Missing required system packages: libncursesw5-dev

To install missing packages, run:
  make install-system
```

**Solution:** Run `make install-system` to install the missing packages, then retry `make install`.

**Verify system packages without installing:**
```bash
make check-system
```

## Documentation

### Core Documentation
- [Technical Decisions](docs/DECISIONS.md) - Decision log with rationale (D-001 to D-007)
- [System Architecture](docs/ARCHITECTURE.md) - Component design and data flow
- [Model Documentation](docs/MODEL.md) - Performance analysis and comparisons

### Project Planning
- [Porting Plan](PLAN_PORTING.md) - Multi-stage implementation roadmap
- [Agent Instructions](AGENTS_PORTING.md) - Code quality guidelines
- [Original Specification](0_OBJECTIVE.md) - Assessment requirements

### Investigations
- [Subcategory Independence Analysis](exploration/subcategory_independence/REPORT.md) - Statistical evidence for scope decisions
- [Exploration Methodology](exploration/README.md) - Investigation approach
