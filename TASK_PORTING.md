# TOODLE Porting Task

## What
Port the DOODLE intelligent support ticket system to TOODLE, cleaning every file
during transfer. The goal is a lean codebase that demonstrates Full-Stack AI Engineer
skills in a one-work-day scope.

## Why
DOODLE grew to ~10,200 lines with experimental code, deferred features still in config,
and LLM-typical over-robustness. The assessment values "demonstrating your approach
rather than perfect implementation of every feature." A clean 4,500-line codebase
demonstrates better engineering judgment than a bloated 10,000-line one.

## Source
- Source: ../DOODLE/ (read-only reference)
- Target: . (active development)
- Agent instructions: AGENTS_PORTING.md

## Stages
0. Scaffold & Config
1. Data Pipeline (JSON → DuckDB → dbt → split parquet)
2. Feature Engineering (TF-IDF + tabular features)
3. Traditional ML (CatBoost + XGBoost + evaluation + MLflow)
4. Deep Learning (DistilBERT + comparison report)
5. Sentiment, Search & Anomaly
6. API & Integration (FastAPI serving)
7. Documentation & Final Polish

## Key Design Decisions (vs DOODLE)
- **Clean training data as primary**: ALL models train on ~110 deduplicated subjects
  (not noisy 100K). This fixes the data quality issue and should achieve >85% F1.
- **LightGBM dropped**: CatBoost + XGBoost + BERT = 3 models to compare (sufficient)
- **Priority kept as placeholder**: /predict returns category (model) + priority/sentiment (placeholders)
- **Subcategory code removed**: 5 feature configs, deferred and never shipped
- **Experimental scripts removed**: ~2,400 lines of one-off analyses
- **Full 100K still flows through dbt**: Demonstrates data engineering, serves RAG corpus

## Per-Stage Protocol
1. Port files, inspecting and cleaning each one
2. Run tests (new + regression)
3. Review significant changes with user
4. User commits

## Key Reference Files (in DOODLE)
- `0_OBJECTIVE.md` — assessment specification
- `0_jobspec_doodle.txt` — evaluated skills
- `CONTEXT.md` — project status and constraints
- `docs/DECISIONS.md` — design rationale
- `prv/DATA.md` — field classifications (leakage rules)
