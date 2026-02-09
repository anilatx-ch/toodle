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

### ✅ Stages 0-2: Complete

- **Stage 0:** Project scaffold, configuration (D-001 to D-005)
- **Stage 1:** Data pipeline, dbt, clean training set (D-006, D-007)
- **Stage 2:** Feature engineering, TF-IDF, categorical encoding (D-008, D-009)

See git history and `docs/DECISIONS.md` for details.

---

### Stage 2.5: Evaluation & Experiment Tracking Infrastructure

**Goal**: Set up shared evaluation and MLflow infrastructure used by both traditional ML (Stage 3) and BERT (Stage 4).

This stage extracts evaluation components that both model training stages need, enabling parallel development.

**Port**:
- `src/evaluation/metrics.py` (~97L) - F1, accuracy, classification reports
- `src/evaluation/analysis.py` (~86L) - confusion matrices, per-class analysis
- `src/evaluation/latency.py` (~39L) - inference latency profiling
- `src/mlflow_utils.py` (~131L) - MLflow logging utilities
- `scripts/run_evaluation.py` - evaluation orchestration

**Tests**:
- `test_metrics.py` - test metric calculations
- `test_evaluation.py` - test evaluation workflow with dummy predictions

**Makefile targets**: `evaluate`

#### Documentation Deliverables (Stage 2.5)

**Create `docs/MODEL.md`** initial structure:

```markdown
# Model Documentation

## Category Classification

### Problem Definition
Predict support ticket category from ticket text and metadata.
- **Classes**: 5 categories (Technical Issue, Billing, Account, Feature Request, General Inquiry)
- **Training data**: ~110 deduplicated subject→category pairs (balanced ~22 per class)
- **Evaluation**: Weighted F1 score (target: >85%)

### Model Comparison
[To be filled by Stages 3, 4, and 4.5]
```

**Append to `docs/DECISIONS.md`**:

```markdown
## D-010: Evaluation Metrics
**Decision**: Weighted F1 as primary metric, with per-class precision/recall
**Rationale**: Balanced dataset (~22 per class) makes weighted F1 appropriate. Per-class metrics help identify category-specific issues.
**Trade-off**: Standard metrics; no custom business-weighted scoring.

## D-011: MLflow Experiment Structure
**Decision**: Single experiment per model type, runs tagged by data version and config hash
**Rationale**: Simple structure for assessment scope. Tags enable filtering without complex experiment hierarchy.
**Trade-off**: Less organization than production MLOps, appropriate for demo.
```

**Update `CONTEXT.md`**:
- Mark Stage 2.5 as ✅ complete
- Update "Current Stage" to reflect parallel Stages 3 & 4 as next
- Update test counts and source LOC count

---

### Stage 3: Traditional ML Training (CAN RUN IN PARALLEL WITH STAGE 4)

**Goal**: Train CatBoost and XGBoost on clean ~110 samples. Expected >85% F1.

All training uses the **clean deduplicated dataset** from Stage 1 (not the noisy 100K).

**Port**:
- `src/models/catboost_model.py` (~123L)
- `src/models/xgboost_model.py` (~166L)
- `src/training/train_catboost.py` — trim (~373→~250L)
- `src/training/train_xgboost.py` — trim (~372→~250L)
- `scripts/run_training.py` — CatBoost/XGBoost parts only

**DO NOT port**: lightgbm_model.py, train_lightgbm.py

**Key cleanups**:
- Training loads clean split parquet (not full noisy parquet)
- Hyperparams tuned for ~110 samples (fewer iterations, appropriate regularization)
- Remove subcategory-specific summary path logic
- Tighten MLflow tagging (remove redundant tags)
- Optuna search space reduced for small dataset (10-20 trials)

**Makefile targets**: `train-catboost`, `train-xgboost`
**Tests**: test_catboost.py, test_xgboost.py, test_run_training.py

**Dependencies**: Stage 1 (data) + Stage 2 (features) + Stage 2.5 (evaluation)

#### Documentation Deliverables (Stage 3)

**Update `docs/MODEL.md`** - add traditional ML sections:

```markdown
### Traditional ML Models

#### CatBoost
- **Architecture**: Gradient boosting with native categorical handling
- **Features**: TF-IDF (5K) + categorical encodings + numerical features
- **Hyperparameters**: [Logged in MLflow]
- **Performance**: F1=X.XX, Accuracy=X.XX, Latency p50=X ms

#### XGBoost
- **Architecture**: Gradient boosting with histogram-based splits
- **Features**: Same as CatBoost
- **Hyperparameters**: [Logged in MLflow]
- **Performance**: F1=X.XX, Accuracy=X.XX, Latency p50=X ms
```

**Append to `docs/DECISIONS.md`**:

```markdown
## D-013: Hyperparameter Tuning Strategy
**Decision**: Optuna with 10-20 trials for small dataset
**Rationale**: With ~110 training samples, extensive tuning risks overfitting. Reduced trials find reasonable hyperparameters without exhaustive search.
**Trade-off**: May miss optimal configuration, but prevents over-tuning.
```

**Update `docs/ARCHITECTURE.md`**: Add Traditional ML Training section.

**Update `CONTEXT.md`**:
- Mark Stage 3 as ✅ complete
- Update test counts and source LOC count

---

### Stage 4: Deep Learning Training (CAN RUN IN PARALLEL WITH STAGE 3)

**Goal**: Train DistilBERT on clean ~110 samples (text-only mode). Expected >85% F1.

BERT trains on same clean dataset as trad ML but uses **raw text directly** (no Stage 2 features needed for text-only mode). The BERT_CLEAN config params become standard BERT params.

**Port**:
- `src/models/bert_model.py` — **significant cleanup** (831→~400L)
- `src/training/train_bert.py` — **rewritten** (~300L, absorb train_bert_category_clean.py approach)

**DO NOT port as separate file**: train_bert_category_clean.py (its approach IS the main training now)

**Key cleanups in bert_model.py**:
- Simplify XLA/CUDA configuration (remove unnecessary workarounds)
- Streamline text-only vs text+tabular branches (keep both but cleaner)
- Remove excessive safety checks around KerasNLP imports
- Clean up session cleanup logic

**Key cleanups in train_bert.py**:
- Primary path trains on clean deduplicated data (no separate "clean" mode)
- BERT_CLEAN params become standard BERT params in config
- ValF1Callback from train_bert_category_clean.py integrated as standard
- Align MLflow logging style with trad ML scripts

**Makefile targets**: `train-bert`, `download-bert`
**Tests**: test_bert_model.py

**Dependencies**: Stage 1 (data) + Stage 2.5 (evaluation)
**Note**: Does NOT depend on Stage 2 (features) for text-only mode

#### Documentation Deliverables (Stage 4)

**Update `docs/MODEL.md`** - add BERT section:

```markdown
### Deep Learning Model

#### DistilBERT
- **Architecture**: DistilBERT base, fine-tuned classification head
- **Input**: Raw text (tokenized by BERT tokenizer)
- **Hyperparameters**: batch_size=16, epochs=4, patience=2, lr=2e-5
- **Performance**: F1=X.XX, Accuracy=X.XX, Latency p50=X ms
```

**Append to `docs/DECISIONS.md`**:

```markdown
## D-014: BERT Training Configuration
**Decision**: Small batch (16), few epochs (4), early stopping (patience=2)
**Rationale**: With ~110 training samples, BERT can overfit quickly. Small batches provide more gradient updates per epoch; early stopping prevents overfitting.
**Trade-off**: May underfit if data is complex, but clean labels make task straightforward.

## D-015: Text-Only vs Multimodal BERT
**Decision**: Text-only as primary path, multimodal as option
**Rationale**: For category classification, subject text is highly predictive (deterministic mapping). Tabular features add complexity without improving clean-data accuracy.
**Trade-off**: Unused multimodal code path, but demonstrates capability.
```

**Update `CONTEXT.md`**:
- Mark Stage 4 as ✅ complete
- Update test counts and source LOC count

---

### Stage 4.5: Model Comparison & Reporting

**Goal**: Generate comprehensive comparison of all three models (CatBoost, XGBoost, BERT).

This stage requires both Stage 3 and Stage 4 to be complete (all models trained).

**Port**:
- `scripts/generate_report.py` — ML comparison report generator

**Makefile targets**: `report`

**Dependencies**: Stage 3 (CatBoost/XGBoost trained) + Stage 4 (BERT trained)

#### Documentation Deliverables (Stage 4.5)

**Update `docs/MODEL.md`** - complete comparison table:

```markdown
### Final Model Comparison

| Model      | F1 Score | Accuracy | Latency (p50) | Latency (p99) | Size (MB) |
|------------|----------|----------|---------------|---------------|-----------|
| CatBoost   | X.XX     | X.XX     | X ms          | X ms          | X         |
| XGBoost    | X.XX     | X.XX     | X ms          | X ms          | X         |
| DistilBERT | X.XX     | X.XX     | X ms          | X ms          | X         |

### Recommendation
[Analysis of which model to use in production and why - consider accuracy vs latency vs size trade-offs]
```

**Update `CONTEXT.md`**:
- Mark Stage 4.5 as ✅ complete
- Update "Current Stage" to reflect Stage 5 as next

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
## D-016: RAG Implementation
**Decision**: FAISS vector search + entity keyword matching (hybrid)
**Rationale**: Assessment requires "RAG + Graph-RAG". FAISS provides fast semantic search; entity extraction enables keyword/error-code matching. Hybrid approach balances recall and precision.
**Trade-off**: Not a full knowledge graph, but demonstrates the pattern.

## D-017: Anomaly Detection Scope
**Decision**: Volume-based anomaly detection on category distribution
**Rationale**: Assessment requires "detecting emerging issues". Category volume shifts indicate new issue types or outages. Simple statistical approach (z-score on rolling windows) is interpretable and debuggable.
**Trade-off**: Won't catch subtle semantic shifts, but covers major incidents.

## D-018: Sentiment Model Backend
**Decision**: Single backend (CatBoost) for sentiment
**Rationale**: Sentiment is secondary to category classification. One well-tuned model suffices for demo. Reduces training time and model storage.
**Trade-off**: No sentiment model comparison, but demonstrates the capability.
```

**Update `docs/ARCHITECTURE.md`**: Add Retrieval and Anomaly sections.

**Update `CONTEXT.md`**:
- Mark Stage 5 as ✅ complete
- Update "Current Stage" to reflect Stage 6 as next
- Update test counts and source LOC count

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
## D-019: Placeholder Fields in API Response
**Decision**: Return priority and sentiment as placeholders with warning flags
**Rationale**: API contract shows intended multi-output design. Clients can integrate now and receive real predictions when models are added. Warning flags make placeholder status explicit.
**Trade-off**: Slightly confusing API, but demonstrates extensible design.

## D-020: Model Ensemble Strategy
**Decision**: Confidence-weighted voting across CatBoost, XGBoost, BERT
**Rationale**: Ensemble reduces variance and catches cases where one model fails. Confidence weighting trusts more certain predictions.
**Trade-off**: Higher latency (3 model calls), but better reliability.
```

**Update `docs/ARCHITECTURE.md`**: Add API Layer section with endpoint diagram.
**Update `README.md`**: Add API usage examples.

**Update `CONTEXT.md`**:
- Mark Stage 6 as ✅ complete
- Update "Current Stage" to reflect Stage 7 as next (final stage)
- Update test counts, source LOC count, and API status

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
- `CONTEXT.md` — **final review: ensure Stage 7 marked complete, all sections accurate**

**Update `CONTEXT.md` at completion**:
- Mark Stage 7 as ✅ complete
- Verify all metrics (test counts, source LOC, performance) are up-to-date
- Confirm all cross-references to other docs are valid

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
