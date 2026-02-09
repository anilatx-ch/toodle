# TOODLE Porting Plan (Improved)

## Context

The DOODLE project (a Full-Stack AI Engineer technical assessment for Doodle) has grown to ~10,200 lines of source code across 40+ modules, with significant bloat from:

- **Deferred features still in code**: subcategory configs (5 duplicate feature specs), priority placeholders, BERT_CLEAN params, backward-compatibility aliases
- **Experimental code left in src/**: deterministic_mappings.py (442L), single_feature_power.py (479L), clean_splitter.py (164L), clean_subject_extractor.py (264L), minimal_pipeline.py (103L), train_bert_category_clean.py (462L)
- **Redundant 3rd traditional ML backend**: LightGBM duplicates what CatBoost+XGBoost already demonstrate
- **Overly defensive code**: verbose error paths for impossible scenarios, excessive comments explaining obvious logic, legacy fallbacks that no codepath uses
- **splitter.py at 565L**: contains template helpers, date-boundary split, resampling, and template validation that are unused experiments

The ported TOODLE project should demonstrate the same skills (see `0_jobspec_doodle.txt`) in a codebase that's clean enough to read in one sitting and clearly the work of a thoughtful engineer, not an LLM transcript.

**Target**: /home/ai_agent/DOODLE/TOODLE/ (existing git repo with context files)

**Estimated reduction**: ~10,200L → ~4,500L source (56% smaller)

---

## Documentation Strategy (NEW)

**Principle**: Document decisions WHEN MADE, not retrospectively in a final stage.

Per 0_OBJECTIVE.md deliverables:
1. **Architecture Documentation** — technology choices with justifications, component interactions
2. **Model Documentation** — performance benchmarks, comparisons, feature importance
3. **README** — setup, API docs, key design decisions

**Implementation**:
- `docs/DECISIONS.md` — living document, started in Stage 0, appended each stage
- `docs/ARCHITECTURE.md` — started in Stage 1, evolves with each component
- `docs/MODEL.md` — started in Stage 3, updated in Stage 4
- `README.md` — started in Stage 0 (minimal), fleshed out progressively

Each stage section below includes a **Documentation Deliverables** subsection.

---

## Scope Decisions

### KEEP (demonstrates skills the jobspec values)
- **CatBoost + XGBoost** — 2 traditional ML backends (jobspec: "XGBoost, LightGBM"; assessment: "XGBoost/CatBoost"). Two is enough to show comparison.
- **DistilBERT** — deep learning comparison (jobspec: "Transformers"; assessment: "TensorFlow/Keras")
- **dbt-DuckDB pipeline** — data engineering (jobspec: "dbt")
- **MLflow tracking** — experiment management (jobspec: "MLflow")
- **FAISS + entity search** — RAG retrieval (assessment: "RAG + Graph-RAG")
- **Anomaly detection** — monitoring/drift awareness (assessment: Phase 3)
- **Sentiment classifier** — multi-task capability (shows Phase 4 delivery)
- **FastAPI serving** — production mindset (assessment: "API endpoints")
- **Docker + Makefile** — reproducibility (assessment: "containerized deployment")
- **Optuna tuning** — hyperparameter optimization (shows ML depth)

### CRITICAL DESIGN CHANGE: Clean Training Data as Primary Pipeline

The DOODLE project trained on 100K noisy samples (30% conflicting labels → 18% F1).
But `subject → category` mapping is **deterministic and clean**: ~110 unique subjects,
each mapping to exactly one category. This makes category classification fundamentally
a **110-sample problem**.

In TOODLE, the clean deduplication approach becomes the **PRIMARY training pipeline**
for ALL models (not a separate "BERT_CLEAN experiment"):

1. Full 100K → dbt pipeline (demonstrates data engineering, used for RAG corpus)
2. Deduplicate: extract unique `subject → category` pairs (~110 samples)
3. Balance: ~22 per category (5 categories)
4. Split: train/val/test stratified by category
5. Train **all** models (CatBoost, XGBoost, BERT) on this clean data
6. Expected: >85% F1 for both trad ML and DL (vs 18%/1.66% on noisy data)

This demonstrates: data quality investigation → solution design → working accuracy.
The `clean_subject_extractor.py` and `clean_splitter.py` logic merges into the main
data pipeline (not kept as separate scripts).

### DROP (no longer in scope)
- **LightGBM** — 3rd trad ML backend is redundant (CatBoost+XGBoost suffice)
- **All subcategory code** — 5 feature configs, model paths, hierarchy decode (deferred, never shipped)
- **Experimental data scripts** — deterministic_mappings, single_feature_power, minimal_pipeline
- **Noisy 100K training path** — replaced by clean deduplication (100K still used for dbt/RAG)
- **3-tier ENV (dev/test/prod)** — simplify to `SMOKE_TEST=true|false` with single `ENV` for path namespacing
- **tools/, adhoc scripts, prv/, .archive_do_not_explore/** — none ported
- **All CSV/diagnostic artifacts** — regenerated, not ported

### KEEP BUT TRANSFORM
- **Priority prediction** — kept as placeholder in /predict API (returns "medium" deterministically). Shows intended multi-output design; real implementation deferred.
- **clean_subject_extractor logic** — merged into main `src/data/splitter.py` (not a separate module)
- **clean_splitter logic** — merged into main `src/data/splitter.py`
- **BERT config params** — the BERT_CLEAN_* params become the standard BERT params (they're tuned for the correct ~110 sample dataset)

### SIMPLIFY (port but significantly leaner)
- **config.py**: 659→~280L — remove subcategory specs, LightGBM, backward-compat aliases; BERT_CLEAN params become standard BERT params
- **splitter.py**: 565→~200L — merge clean_subject_extractor + clean_splitter + stratified split into one coherent module; drop template helpers, date_boundary_split, legacy time-based split
- **app.py**: 477→~350L — remove LightGBM loading; keep priority as placeholder; simplify ModelManager
- **bert_model.py**: 831→~400L — simplify XLA/device handling, reduce conditional branches
- **train_*.py**: each ~370→~250L — tighten, train on clean ~110 samples by default
- **preprocessing.py**: 279→~150L — remove unused augmentation paths

---

## Stage Definitions

**Pre-stage**: Save this plan as `TOODLE/PLAN_PORTING.md` (first action upon execution).

Each stage ends with: tests green, documentation updated, review with user, user runs git commit.

---

### Stage 0: Scaffold & Config

**Goal**: Bootable project skeleton that imports and passes config tests.

**Create/port**:
- `pyproject.toml` — cleaned deps (drop LightGBM, consolidate groups)
- `Makefile` — skeleton with install, data-pipeline, test targets (no training yet)
- `.gitignore` — from DOODLE, tuned
- `Dockerfile`, `docker-compose.yml` — cleaned
- `src/__init__.py`
- `src/config.py` — **heavily cleaned** (see Scope Decisions above)
- `tests/test_config.py`
- `AGENTS.md` — agent instructions for this project

**Key cleanups in config.py**:
- Remove all subcategory feature configs and paths
- Remove LightGBM params/paths/search spaces
- BERT_CLEAN_* params become the standard BERT_* params (rename, remove duplicate block)
- Remove backward-compatible aliases (FEATURE_PIPELINE_PATH, CATBOOST_MODEL_PATH, etc.)
- Keep priority in MULTI_CLASSIFIER_ORDER (as placeholder target)
- Collapse IS_INFRA_DEV/SMOKE_TEST triple-branch into cleaner 2-level
- Remove SUBCATEGORY_CLASSES
- Add CLEAN_TRAINING_PARQUET_PATH for deduplicated training data
- Slim down ensure_directories (just create what's needed)

**Tests**: config imports, ENV handling, paths resolve

#### Documentation Deliverables (Stage 0)

**Create `docs/DECISIONS.md`** with initial entries:

```markdown
# Technical Decisions Log

## D-001: Traditional ML Framework Selection
**Decision**: CatBoost + XGBoost (drop LightGBM)
**Rationale**: Two gradient boosting implementations suffice to demonstrate comparison skills. LightGBM adds marginal value while increasing maintenance surface. CatBoost offers native categorical handling; XGBoost is industry standard.
**Trade-off**: Less framework breadth vs cleaner codebase.

## D-002: Deep Learning Framework
**Decision**: DistilBERT via keras-nlp on TensorFlow
**Rationale**: Assessment specifies "TensorFlow/Keras". DistilBERT balances accuracy with inference speed for a demo system.
**Trade-off**: Heavier dependency than sklearn, but demonstrates transformer competency.

## D-003: Data Pipeline Stack
**Decision**: dbt + DuckDB (in-process)
**Rationale**: Job spec values dbt experience. DuckDB avoids external database dependency while supporting SQL transforms. Demonstrates data engineering without infrastructure overhead.
**Trade-off**: Not production-scale, but appropriate for assessment scope.

## D-004: Experiment Tracking
**Decision**: MLflow local tracking
**Rationale**: Job spec explicitly lists MLflow. Local file store avoids server dependency.
**Trade-off**: No remote collaboration features, acceptable for single-developer demo.

## D-005: Environment Simplification
**Decision**: SMOKE_TEST boolean + single ENV variable
**Rationale**: Original 3-tier (dev/test/prod) added complexity without value in assessment context. SMOKE_TEST controls data size; ENV controls path namespacing.
**Trade-off**: Less production-realistic, but cleaner for demo purposes.
```

**Create `README.md`** (minimal, to be expanded):

```markdown
# TOODLE - Intelligent Support Ticket System

Technical assessment demonstrating ML pipeline, NLP classification, and production API design.

## Quick Start
```bash
make install
make all SMOKE_TEST=true  # Full pipeline in smoke mode
make api                   # Start API server
```

## Project Status
Stage 0 complete. See docs/DECISIONS.md for technology choices.

## Documentation
- [Technical Decisions](docs/DECISIONS.md)
```

---

### Stage 1: Data Pipeline

**Goal**: JSON → DuckDB → dbt → deduplicated clean training parquet + full corpus, validated.

The pipeline has TWO output paths:
1. **Full corpus** (100K records via dbt) — used for RAG/search, embeddings, anomaly baselines
2. **Clean training set** (~110 deduplicated subject→category pairs, balanced) — used for ALL model training

**Port (merged/rewritten)**:
- `src/data/__init__.py`
- `src/data/loader.py` (~70L, mostly clean — minor tidying)
- `src/data/splitter.py` — **rewritten** (merge clean_subject_extractor + clean_splitter + stratified split into one module, ~200L)
- `src/data/schema.py` (~63L, clean)
- `dbt_project/` — models, profiles, sources (clean, minimal)
- `src/models/hierarchy.py` (~114L — still needed for derive_category)

**Key design of new splitter.py**:
- `extract_clean_subjects(df) → df` — deduplicate full dataset to ~110 unique subject→category pairs (logic from clean_subject_extractor.py)
- `balance_categories(df, target_per_category) → df` — balance to equal representation per category (logic from clean_subject_extractor.py)
- `stratified_split(df) → df` — 70/15/15 train/val/test with stratification by category
- `main()` — orchestrates: load full → dbt → extract clean → balance → split → save
- Drop: template_id, error_code extraction, date_boundary_split, resample_train_set, validate_template_consistency, legacy time-based split

**DO NOT port as separate files**: clean_splitter.py, clean_subject_extractor.py, deterministic_mappings.py, single_feature_power.py (the useful logic from the first two is merged into splitter.py)

**Makefile targets**: `data-pipeline`, `dbt-run`, `dbt-test`
**Tests**: test_data_pipeline.py (adapted), test_splitter.py (test deduplication, balance, split proportions), test_hierarchy.py

#### Documentation Deliverables (Stage 1)

**Append to `docs/DECISIONS.md`**:

```markdown
## D-006: Clean Training Data Strategy
**Decision**: Train all models on ~110 deduplicated subject→category pairs, not noisy 100K
**Rationale**: Investigation revealed subject→category mapping is deterministic (~110 unique pairs). The 100K dataset has ~30% label noise from the generation process. Training on clean data is the correct approach; noisy training was a data quality bug, not a feature.
**Trade-off**: Smaller training set, but correct labels. Expected: >85% F1 vs 18% on noisy data.
**Evidence**: See diagnostics/data_investigation/ for analysis.

## D-007: Dual Output Pipeline
**Decision**: dbt processes full 100K (for RAG corpus), splitter extracts clean ~110 (for training)
**Rationale**: Full corpus needed for semantic search and anomaly baselines. Clean subset needed for accurate classification training. Both derive from same source with different purposes.
**Trade-off**: Two data paths to maintain, but each serves distinct purpose.
```

**Create `docs/ARCHITECTURE.md`** (initial):

```markdown
# System Architecture

## Overview

TOODLE is an intelligent support ticket system with three main subsystems:
1. **Data Pipeline** — ingestion, transformation, and splitting
2. **ML Models** — classification (category, priority, sentiment)
3. **Serving Layer** — FastAPI endpoints for prediction and search

## Data Flow

```
tickets.json (100K)
       │
       ▼
   DuckDB (staging)
       │
       ▼
   dbt transforms
       │
       ├──────────────────┐
       ▼                  ▼
 Full Corpus (100K)   Clean Training (~110)
       │                  │
       ▼                  ▼
 RAG/Search Index    ML Model Training
 Anomaly Baselines   (CatBoost, XGBoost, BERT)
```

## Component Details

### Data Pipeline (Stage 1)
- **loader.py**: JSON ingestion to DuckDB
- **splitter.py**: Deduplication, balancing, train/val/test split
- **dbt_project/**: SQL transformations for corpus preparation

[To be expanded in subsequent stages]
```

**Update `README.md`**: Add "Data Pipeline" section with basic usage.

---

### Stage 2: Feature Engineering

**Goal**: Feature pipeline produces TF-IDF + tabular matrices from split parquet.

**Port**:
- `src/features/__init__.py`
- `src/features/pipeline.py` — clean (remove subcategory spec references, simplify)
- `src/features/text.py` (~53L, clean)
- `src/features/categorical.py` (~55L, clean)
- `src/features/numerical.py` (~63L, clean)
- `src/features/preprocessing.py` — trim unused augmentation (279→~150L)
- `src/features/entities.py` (~172L, used by search — port as-is or light trim)
- `scripts/run_features.py`

**DO NOT port**: minimal_pipeline.py

**Key cleanups**:
- pipeline.py: Remove `create_multi_classifier_pipelines` factory (only needed for deferred multi-classifier batch)
- preprocessing.py: Review each augmentation; remove any that feed only dropped classifiers

**Makefile targets**: `features`
**Tests**: test_features.py, test_preprocessing.py (trimmed), test_entities.py

#### Documentation Deliverables (Stage 2)

**Append to `docs/DECISIONS.md`**:

```markdown
## D-008: Feature Engineering Approach
**Decision**: TF-IDF for text + one-hot encoding for categoricals + standard scaling for numericals
**Rationale**: Simple, interpretable features that work well with gradient boosting. TF-IDF captures term importance without embedding overhead. Categorical encoding preserves interpretability.
**Trade-off**: Less semantic richness than embeddings, but faster and more debuggable.

## D-009: Feature Pipeline Simplification
**Decision**: Single unified pipeline (remove multi-classifier factory)
**Rationale**: Original design anticipated multiple classifier targets sharing feature pipelines. With only category as real target (priority/sentiment are placeholders), the factory pattern adds complexity without benefit.
**Trade-off**: Less flexibility for future expansion, but cleaner current code.
```

**Update `docs/ARCHITECTURE.md`**: Add Feature Engineering section.

---

### Stage 3: Traditional ML Models + Evaluation

**Goal**: CatBoost and XGBoost train on clean ~110 samples, evaluate, save artifacts, log to MLflow.

All training uses the **clean deduplicated dataset** from Stage 1 (not the noisy 100K).
With clean labels, these models should achieve **>85% F1** on category prediction.

**Port**:
- `src/models/catboost_model.py` (~123L, clean)
- `src/models/xgboost_model.py` (~166L, clean)
- `src/training/train_catboost.py` — trim (~373→~250L)
- `src/training/train_xgboost.py` — trim (~372→~250L)
- `src/evaluation/metrics.py` (~97L)
- `src/evaluation/analysis.py` (~86L)
- `src/evaluation/latency.py` (~39L)
- `src/mlflow_utils.py` (~131L, clean)
- `scripts/run_training.py` — simplified (no LightGBM)
- `scripts/run_evaluation.py`

**DO NOT port**: lightgbm_model.py, train_lightgbm.py

**Key cleanups in training scripts**:
- Training loads the clean split parquet (not the full noisy parquet)
- Hyperparams tuned for ~110 samples (fewer iterations, appropriate regularization)
- Remove subcategory-specific summary path logic
- Tighten MLflow tagging (remove redundant tags)
- Consider extracting shared `_prepare_splits` / `_load_or_fit_pipeline` / `_log_metrics` into a small `training/_common.py` if it reduces net lines (only if it actually helps)
- Optuna search space reduced for small dataset (fewer trials needed)

**Makefile targets**: `train-catboost`, `train-xgboost`, `evaluate`, `report`
**Tests**: test_catboost.py, test_xgboost.py, test_evaluation.py, test_run_training.py

#### Documentation Deliverables (Stage 3)

**Create `docs/MODEL.md`**:

```markdown
# Model Documentation

## Category Classification

### Problem Definition
Predict support ticket category from ticket text and metadata.
- **Classes**: 5 categories (Technical Issue, Billing, Account, Feature Request, General Inquiry)
- **Training data**: ~110 deduplicated subject→category pairs (balanced ~22 per class)
- **Evaluation**: Weighted F1 score (target: >85%)

### Traditional ML Models

#### CatBoost
- **Architecture**: Gradient boosting with native categorical handling
- **Features**: TF-IDF text vectors + categorical encodings + numerical features
- **Hyperparameters**: [Tuned via Optuna, logged in MLflow]
- **Performance**: [F1: X.XX, Accuracy: X.XX] ← filled after training

#### XGBoost
- **Architecture**: Gradient boosting with histogram-based splits
- **Features**: Same as CatBoost
- **Hyperparameters**: [Tuned via Optuna, logged in MLflow]
- **Performance**: [F1: X.XX, Accuracy: X.XX] ← filled after training

### Model Comparison
| Model    | F1 Score | Accuracy | Latency (p50) | Latency (p99) |
|----------|----------|----------|---------------|---------------|
| CatBoost | X.XX     | X.XX     | X ms          | X ms          |
| XGBoost  | X.XX     | X.XX     | X ms          | X ms          |

[To be completed after training runs]
```

**Append to `docs/DECISIONS.md`**:

```markdown
## D-010: Hyperparameter Tuning Strategy
**Decision**: Optuna with reduced trial count for small dataset
**Rationale**: With ~110 training samples, extensive tuning risks overfitting. Fewer trials (10-20) find reasonable hyperparameters without exhaustive search.
**Trade-off**: May miss optimal configuration, but prevents over-tuning to small dataset.

## D-011: MLflow Experiment Structure
**Decision**: Single experiment per model type, runs tagged by data version and config hash
**Rationale**: Simple structure for assessment scope. Tags enable filtering without complex experiment hierarchy.
**Trade-off**: Less organization than production MLOps, appropriate for demo.
```

**Update `docs/ARCHITECTURE.md`**: Add ML Training section.

---

### Stage 4: Deep Learning (BERT)

**Goal**: DistilBERT trains on clean ~110 samples, evaluates, produces comparison with traditional ML.

BERT trains on the same clean dataset as trad ML. The BERT_CLEAN config params
(batch_size=16, epochs=4, patience=2) become the standard BERT config — they're
already tuned for this ~110-sample regime. Expected: >85% F1 (DOODLE achieved ~1.0 F1
on clean data with BERT).

**Port**:
- `src/models/bert_model.py` — **significant cleanup** (831→~400L)
- `src/training/train_bert.py` — **rewritten** to absorb the clean-data training approach from `train_bert_category_clean.py` as the primary path (~300L)
- `scripts/generate_report.py` — ML comparison report

**DO NOT port as separate file**: train_bert_category_clean.py (its approach IS the main training now)

**Key cleanups in bert_model.py**:
- Simplify XLA/CUDA configuration (remove workarounds that may not be needed)
- Reduce conditional branches for text-only vs text+tabular (keep both but streamline)
- Remove excessive safety checks around KerasNLP imports
- Clean up session cleanup logic

**Key cleanups in train_bert.py**:
- Primary path trains on clean deduplicated data (no separate "clean" mode)
- BERT_CLEAN params become standard BERT params in config
- ValF1Callback from train_bert_category_clean.py integrated as standard
- Simplify epoch/callback setup
- Align MLflow logging style with trad ML scripts

**Makefile targets**: `train-bert`, `download-bert`
**Tests**: test_bert_model.py (adapted)

#### Documentation Deliverables (Stage 4)

**Update `docs/MODEL.md`**: Add BERT section and complete comparison table:

```markdown
#### DistilBERT
- **Architecture**: DistilBERT base, fine-tuned classification head
- **Input**: Raw text (tokenized by BERT tokenizer)
- **Hyperparameters**: batch_size=16, epochs=4, patience=2, lr=2e-5
- **Performance**: [F1: X.XX, Accuracy: X.XX] ← filled after training

### Final Model Comparison
| Model      | F1 Score | Accuracy | Latency (p50) | Latency (p99) | Size (MB) |
|------------|----------|----------|---------------|---------------|-----------|
| CatBoost   | X.XX     | X.XX     | X ms          | X ms          | X         |
| XGBoost    | X.XX     | X.XX     | X ms          | X ms          | X         |
| DistilBERT | X.XX     | X.XX     | X ms          | X ms          | X         |

### Recommendation
[Analysis of which model to use in production and why]
```

**Append to `docs/DECISIONS.md`**:

```markdown
## D-012: BERT Training Configuration
**Decision**: Small batch (16), few epochs (4), early stopping (patience=2)
**Rationale**: With ~110 training samples, BERT can overfit quickly. Small batches provide more gradient updates per epoch; early stopping prevents overfitting.
**Trade-off**: May underfit if data is more complex than expected, but clean labels make task straightforward.

## D-013: Text-Only vs Multimodal BERT
**Decision**: Keep text-only as primary path, multimodal as option
**Rationale**: For category classification, subject text is highly predictive (deterministic mapping). Tabular features add complexity without improving clean-data accuracy.
**Trade-off**: Unused multimodal code path, but demonstrates capability if needed.
```

---

### Stage 5: Sentiment, Search & Anomaly

**Goal**: All Phase 2-4 components working: sentiment endpoint, FAISS search, anomaly detection.

**Port**:
- `src/training/train_sentiment.py` — **cleanup** (525→~200L, single-backend focus)
- `src/retrieval/search.py` (~185L)
- `src/retrieval/index.py` (~125L)
- `src/retrieval/embeddings.py` (~108L)
- `src/retrieval/entities.py` (~146L)
- `src/retrieval/corpus.py` (~85L)
- `src/anomaly/detector.py` (~243L, light trim)
- `src/anomaly/baselines.py` (~205L, light trim)
- `src/anomaly/volume_analyzer.py` (~252L, light trim)
- `scripts/build_search_index.py`
- `scripts/generate_embeddings.py`
- `scripts/build_anomaly_baseline.py`

**Key cleanups in train_sentiment.py**:
- Decide if all 3 backends (XGBoost/CatBoost/BERT) are needed for sentiment or if 1 suffices
- Remove redundant backend-selection boilerplate
- Simplify feedback text handling

**Makefile targets**: `train-sentiment`, `build-search-index`, `build-anomaly-baseline`
**Tests**: test_search.py, test_anomaly.py (+ sentiment endpoint tested in Stage 6)

#### Documentation Deliverables (Stage 5)

**Append to `docs/DECISIONS.md`**:

```markdown
## D-014: RAG Implementation
**Decision**: FAISS vector search + entity keyword matching (hybrid)
**Rationale**: Assessment requires "RAG + Graph-RAG". FAISS provides fast semantic search; entity extraction enables keyword/error-code matching. Hybrid approach balances recall and precision.
**Trade-off**: Not a full knowledge graph, but demonstrates the pattern.

## D-015: Anomaly Detection Scope
**Decision**: Volume-based anomaly detection on category distribution
**Rationale**: Assessment requires "detecting emerging issues". Category volume shifts indicate new issue types or outages. Simple statistical approach (z-score on rolling windows) is interpretable and debuggable.
**Trade-off**: Won't catch subtle semantic shifts, but covers major incidents.

## D-016: Sentiment Model Backend
**Decision**: Single backend (CatBoost) for sentiment
**Rationale**: Sentiment is secondary to category classification. One well-tuned model suffices for demo. Reduces training time and model storage.
**Trade-off**: No sentiment model comparison, but demonstrates the capability.
```

**Update `docs/ARCHITECTURE.md`**: Add Retrieval and Anomaly sections.

---

### Stage 6: API & Integration

**Goal**: FastAPI serves /predict, /analyze-feedback, /search, /health. Full smoke pipeline works.

**Prediction scope**: category (real model), priority (placeholder "medium"), sentiment (placeholder "neutral" in /predict; real analysis via /analyze-feedback).

**Port**:
- `src/api/app.py` — **cleaned** (477→~350L)
- `src/api/schemas.py` — **updated** (keep priority/sentiment as placeholders in response)
- `src/api/search.py` (~133L)
- `tests/conftest.py`

**Key cleanups in app.py**:
- Remove LightGBM model loading block
- Keep priority/sentiment as deterministic placeholders in /predict (shows multi-output intent)
- Real sentiment analysis stays at /analyze-feedback endpoint
- Simplify ModelManager.load_models (2 trad ML + BERT, not 3+BERT)
- Simplify _predict_distribution (fewer branches — no LightGBM)
- Cleaner warning flag logic

**Key cleanups in schemas.py**:
- Keep priority/sentiment fields in response (as placeholders with warning flag)
- Simplify naming and contract version

**Makefile targets**: `api`, `docker-build`, `docker-up`, `test`
**Tests**: test_api.py (updated), test_contract_lock.py (updated), full smoke test

#### Documentation Deliverables (Stage 6)

**Create `docs/API_CONTRACT.md`**:

```markdown
# API Contract

## Endpoints

### POST /predict
Predict category for a support ticket.

**Request**:
```json
{
  "subject": "Database sync failing with timeout",
  "description": "Getting ERROR_TIMEOUT_429...",
  "product": "DataSync Pro",
  "customer_tier": "enterprise"
}
```

**Response**:
```json
{
  "category": "Technical Issue",
  "category_confidence": 0.92,
  "priority": "medium",
  "priority_confidence": null,
  "sentiment": "neutral",
  "warnings": ["priority_placeholder", "sentiment_placeholder"]
}
```

### POST /analyze-feedback
Analyze sentiment of feedback text.

### POST /search
Search for similar tickets and solutions.

### GET /health
System health check.

[Full request/response examples for each endpoint]
```

**Append to `docs/DECISIONS.md`**:

```markdown
## D-017: Placeholder Fields in API Response
**Decision**: Return priority and sentiment as placeholders with warning flags
**Rationale**: API contract shows intended multi-output design. Clients can integrate now and receive real predictions when models are added. Warning flags make placeholder status explicit.
**Trade-off**: Slightly confusing API, but demonstrates extensible design.

## D-018: Model Ensemble Strategy
**Decision**: Confidence-weighted voting across CatBoost, XGBoost, BERT
**Rationale**: Ensemble reduces variance and catches cases where one model fails. Confidence weighting trusts more certain predictions.
**Trade-off**: Higher latency (3 model calls), but better reliability.
```

**Update `docs/ARCHITECTURE.md`**: Add API Layer section with endpoint diagram.
**Update `README.md`**: Add API usage examples.

---

### Stage 7: Documentation Polish & Final Verification

**Goal**: Ship-ready documentation, final integration test, clean commit history.

**This stage CONSOLIDATES and POLISHES existing docs, not creates from scratch.**

**Polish**:
- `README.md` — ensure complete setup instructions, architecture overview, all examples work
- `docs/ARCHITECTURE.md` — add system diagram, verify all sections complete
- `docs/DECISIONS.md` — review for clarity, ensure all major decisions captured
- `docs/MODEL.md` — fill in all performance numbers from actual training runs
- `docs/API_CONTRACT.md` — verify examples match actual API behavior
- `CONTEXT.md` — update for TOODLE scope (ported from DOODLE, adapted)

**NOT ported**: FUTURE.md (this is the final version), SKILL_TO_SHOW.md (meta-doc), TODO.md, CODE_FLOW.md (redundant with ARCHITECTURE)

**Final verification**:
- `make test` passes
- `make all` completes (smoke mode)
- Docker builds
- API responds correctly to sample requests
- Documentation accurately reflects code
- All cross-references in docs are valid

---

## AGENTS.md Content

To be written to TOODLE at Stage 0. Full content:

```markdown
# TOODLE Agent Instructions

## Task Context
You are working on TOODLE, a cleaned and focused version of the DOODLE intelligent support ticket system. This is a technical assessment demonstrating Full-Stack AI Engineer skills.

## Quality Bar
Code must demonstrate: clean design, practical trade-offs, production thinking. It is a one-work-day demo, not a production system.

## The Core Question
For every function, class, config block, and error handler, ask:
**"Could this be done cleaner or leaner while maintaining clarity?"**

## What To Cut
1. Defensive code for impossible scenarios
2. Backward-compatibility shims (no old version exists)
3. Comments that restate the code
4. Verbose error messages with usage instructions
5. Excessive logging (one info log per major operation)
6. Dead code paths
7. Over-parameterized functions

## What To Keep
1. Type hints
2. Dataclasses for structured data
3. MLflow logging (demonstrates MLOps)
4. Error handling at system boundaries
5. Deterministic seeds
6. The actual ML logic

## Scope Boundaries
- Two traditional ML backends: CatBoost + XGBoost
- One deep learning backend: DistilBERT
- /predict returns: category (model) + priority/sentiment (placeholders)
- Clean training data: ~110 deduplicated samples
- Full 100K through dbt for RAG corpus

## Documentation Protocol
- Update docs/DECISIONS.md when making significant choices
- Keep docs/ARCHITECTURE.md current with structural changes
- Update docs/MODEL.md with performance numbers after training

## Git Protocol
- Agent suggests commit commands; user executes them
- One commit per stage
```

---

## Verification Plan

After all stages complete:

1. **Unit tests**: `make test` — all pass in smoke mode
2. **Pipeline smoke**: `make all ENV=dev SMOKE_TEST=true` — full pipeline completes
3. **API smoke**:
   - `make api` starts without errors
   - `POST /predict` returns category prediction + priority/sentiment placeholders
   - `POST /analyze-feedback` returns sentiment
   - `POST /search` returns results
   - `GET /health` shows all components ready
4. **Docker**: `make docker-build && make docker-up` — container starts and serves
5. **Documentation**: README accurately describes setup and usage; all referenced files exist; docs/DECISIONS.md has 15+ entries
6. **Line count**: `wc -l src/**/*.py` confirms ~4,500L target (±20%)
