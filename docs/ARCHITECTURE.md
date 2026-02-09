# System Architecture

**Purpose:** Technical architecture documentation for TOODLE
**Status:** Living document - updated each stage
**Last Updated:** 2026-02-09

---

## Overview

The core system now includes Stage 1-7 components:
1. **Full corpus** (100K tickets) → RAG, search, anomaly detection
2. **Clean training set** (~110 samples) → Category prediction models
3. **Traditional ML trainers** (CatBoost + XGBoost) with MLflow and evaluation artifacts
4. **Deep learning** (DistilBERT) for text-based category classification
5. **Sentiment classifier** (CatBoost) trained on feedback text
6. **Retrieval and anomaly modules** for semantic search and issue monitoring
7. **FastAPI serving layer** with /predict, /analyze-feedback, /search, /health endpoints

This architecture addresses the data quality challenge: 100K raw tickets contain 30% conflicting labels, while ~110 unique subject templates provide clean, deterministic category mappings.

---

## System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TOODLE System Architecture                        │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: support_tickets.json (100K tickets)
   │
   ├─────────────────────────────────────────────────────────────────────────┐
   │                                                                           │
   v                                                                           v
┌──────────────────────────────┐                      ┌──────────────────────────┐
│  DATA PIPELINE (Stage 1)     │                      │  CLEAN DATA EXTRACTION   │
│  • Loader → DuckDB           │                      │  (Stage 1 - splitter)    │
│  • dbt transformations       │                      │  • Deduplicate subjects  │
│    - stg_tickets            │                      │  • Validate zero         │
│    - mart_tickets_features  │                      │    conflicts             │
│  • Output: featured_tickets │                      │  • Balance & stratify    │
│    (100K with features)     │                      │  • Output: ~110 clean    │
└──────────────────────────────┘                      │    training samples      │
   │                                                   └──────────────────────────┘
   │                                                              │
   ├──────────────────┬──────────────────┐                      │
   v                  v                  v                      v
┌─────────────┐  ┌─────────────┐  ┌──────────────┐    ┌────────────────────┐
│ EMBEDDINGS  │  │  ANOMALY    │  │   CORPUS     │    │ FEATURE PIPELINE   │
│ (Stage 5)   │  │  BASELINE   │  │  (Stage 5)   │    │ (Stage 2)          │
│ • DistilBERT│  │ (Stage 5)   │  │ • Resolution │    │ • TF-IDF (5K)      │
│   CLS       │  │ • Category  │  │   documents  │    │ • Categorical      │
│   vectors   │  │   volume    │  │ • Metadata   │    │   encoding         │
│ • 768-dim   │  │ • Confidence│  │              │    │ • Numerical        │
│   per ticket│  │   stats     │  │              │    │   scaling          │
└─────────────┘  └─────────────┘  └──────────────┘    └────────────────────┘
   │                  │                 │                      │
   v                  │                 v                      v
┌─────────────────┐   │         ┌──────────────┐    ┌────────────────────────┐
│ SEARCH INDEX    │   │         │ ENTITY INDEX │    │ MODEL TRAINING         │
│ (Stage 5)       │   │         │ (Stage 5)    │    │ (Stages 3-4)           │
│ • FAISS         │   │         │ • Error codes│    │                        │
│   IndexFlatIP   │   │         │ • Products   │    │ ┌──────────────────┐   │
│ • ID mapping    │   │         │ • Inverted   │    │ │ CatBoost (S3)    │   │
│                 │   │         │   index      │    │ │ • 123 LOC        │   │
└─────────────────┘   │         └──────────────┘    │ │ • F1: ~0.91      │   │
   │                  │                 │            │ └──────────────────┘   │
   │                  │                 │            │                        │
   │                  │                 │            │ ┌──────────────────┐   │
   │                  │                 │            │ │ XGBoost (S3)     │   │
   │                  │                 │            │ │ • 166 LOC        │   │
   │                  │                 │            │ │ • F1: ~0.89      │   │
   │                  │                 │            │ └──────────────────┘   │
   │                  │                 │            │                        │
   │                  │                 │            │ ┌──────────────────┐   │
   │                  │                 │            │ │ DistilBERT (S4)  │   │
   │                  │                 │            │ │ • 831 LOC        │   │
   │                  │                 │            │ │ • F1: varies     │   │
   │                  │                 │            │ └──────────────────┘   │
   │                  │                 │            │                        │
   │                  │                 │            │ ┌──────────────────┐   │
   │                  │                 │            │ │ Sentiment (S5)   │   │
   │                  │                 │            │ │ • CatBoost       │   │
   │                  │                 │            │ │ • Feedback text  │   │
   │                  │                 │            │ └──────────────────┘   │
   │                  │                 │            └────────────────────────┘
   │                  │                 │                      │
   │                  │                 │                      v
   │                  │                 │            ┌────────────────────────┐
   │                  │                 │            │ MODEL ARTIFACTS        │
   │                  │                 │            │ • models/*.cbm, *.json │
   │                  │                 │            │ • features/*.pkl       │
   │                  │                 │            │ • metrics/*.json       │
   │                  │                 │            └────────────────────────┘
   │                  │                 │                      │
   └──────────────────┴─────────────────┴──────────────────────┘
                                 │
                                 v
              ┌────────────────────────────────────────────┐
              │   FASTAPI SERVING LAYER (Stage 6)         │
              │   • ModelManager (load all models)        │
              │   • FeaturePipeline (transform inputs)    │
              │   • AnomalyDetector (confidence checks)   │
              │   • SearchEngine (hybrid retrieval)       │
              └────────────────────────────────────────────┘
                                 │
                 ┌───────────────┼───────────────┬────────────────┐
                 v               v               v                v
          ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
          │ /predict │   │ /analyze- │   │ /search  │   │ /health  │
          │          │   │ feedback  │   │          │   │          │
          │ Category │   │ Sentiment │   │ RAG      │   │ Status   │
          │ + placeh.│   │ (real)    │   │ retrieval│   │ check    │
          └──────────┘   └───────────┘   └──────────┘   └──────────┘
                                 │
                                 v
                          CLIENT APPLICATIONS
```

**Key Flows:**

1. **Training Flow** (offline):
   - 100K → dbt → clean ~110 → feature pipeline → model training → artifacts

2. **Serving Flow** (online):
   - Request → /predict → feature pipeline → model inference → response

3. **Retrieval Flow** (online):
   - Query → /search → embedding → FAISS + entity boost → results

---

## Detailed Component Diagram

```
support_tickets.json (100K records)
    │
    ├─→ [1] Loader (src/data/loader.py)
    │       ├─ Parse JSON
    │       ├─ Basic validation
    │       └─ Load to DuckDB raw_tickets table
    │
    ├─→ [2] dbt Transformations (dbt_project/)
    │       │
    │       ├─→ stg_tickets.sql
    │       │   ├─ Type casting (TIMESTAMP, DATE)
    │       │   ├─ Basic cleanup (trim, normalize)
    │       │   └─ Output: stg_tickets (100K)
    │       │
    │       └─→ mart_tickets_features.sql
    │           ├─ Feature extraction
    │           │  ├─ error_code (ERROR_* pattern)
    │           │  ├─ template_id (content hash)
    │           │  ├─ product_month (YYYY-MM)
    │           │  └─ clean_text fields
    │           └─ Output: featured_tickets (100K)
    │
    └─→ [3] Clean Data Extractor (src/data/splitter.py)
            │
            ├─→ Extract unique subject→category mappings
            │   ├─ Group by subject
            │   ├─ Validate deterministic mapping (no conflicts)
            │   └─ Result: ~110 unique templates
            │
            ├─→ Balance & Stratify
            │   ├─ Sample ~22 per category (5 categories)
            │   ├─ Stratified split: 70/15/15 (train/val/test)
            │   └─ Maintain category distribution
            │
            └─→ Output: clean_training_dev.parquet (~110 records)
                    ↓
            [4] Model Training (Stage 3-4)
                ├─ CatBoost
                ├─ XGBoost
                └─ BERT
                    ↓
            [5] Retrieval Index Build (Stage 5)
                ├─ DistilBERT embeddings
                ├─ FAISS vector index
                ├─ Entity inverted index
                └─ Resolution corpus snapshot
                    ↓
            [6] Anomaly Baseline Build (Stage 5)
                └─ Category volume/confidence baseline
```

---

## Component Details

### [1] Loader (src/data/loader.py)

**Purpose:** Initial ingestion from JSON to relational storage

**Input:**
- `support_tickets.json` (100K records; root path)
- or `data/raw/tickets.json` (alternate path)

**Processing:**
1. Parse JSON array
2. Basic schema validation (required fields present)
3. Insert into DuckDB `raw_tickets` table

**Output:**
- DuckDB table: `raw_tickets` (100K rows)
- Columns: ticket_id, created_at, subject, description, category, subcategory, error_logs, stack_trace, product_module, customer_sentiment, priority

**Technology:** Python + DuckDB
**Rationale:** Lightweight embedded database, no server required, SQL-compatible

---

### [2] dbt Transformations (dbt_project/)

#### stg_tickets.sql (Staging Layer)

**Purpose:** Type casting and basic cleanup

**Transformations:**
- Parse timestamps: `created_at::TIMESTAMP` → `created_at`, extract `created_date::DATE`
- Text normalization: TRIM whitespace, handle NULL/empty strings
- Preserve all original fields (no filtering)

**Output:** `stg_tickets` table (100K rows)

**Why separate staging?** Isolates type casting from business logic, enables incremental testing

---

#### mart_tickets_features.sql (Feature Mart)

**Purpose:** Derive analytical features for models and analysis

**Features Extracted:**

1. **error_code** (VARCHAR)
   - Extract first `ERROR_[A-Z0-9_]+` pattern from description
   - NULL if no error code present
   - Purpose: Categorical signal for error classification

2. **template_id** (INTEGER)
   - Hash of (subject, description, error_logs, stack_trace, product_module)
   - Groups identical content regardless of labels
   - Purpose: Identify duplicate/near-duplicate tickets

3. **product_month** (VARCHAR)
   - Extract YYYY-MM from `created_at`
   - Purpose: Temporal analysis, detect trends

4. **clean_subject, clean_description** (VARCHAR)
   - Original text (no masking in dbt, keep raw for RAG)
   - Purpose: Full-text search, embedding generation

**Output:** `featured_tickets` table (100K rows)

**Technology:** dbt + DuckDB
**Rationale:** SQL-based transformations are version-controlled, testable, and auditable

---

### [3] Clean Data Extractor (src/data/splitter.py)

**Purpose:** Extract clean training set from noisy 100K corpus

#### Step 3.1: Extract Unique Templates

```python
# Group by subject, check for conflicting labels
subject_groups = df.groupby('subject')['category'].agg(['unique', 'count'])
conflict_subjects = subject_groups[subject_groups['unique'].apply(len) > 1]

if len(conflict_subjects) > 0:
    raise ValueError(f"Found {len(conflict_subjects)} subjects with multiple categories")
```

**Result:** ~110 unique subject → category mappings (zero conflicts)

**Why this works:** Subject→category mapping is **deterministic** in the data. Multiple tickets with identical subjects always have the same category (proven in data investigation).

---

#### Step 3.2: Balance & Stratify

```python
# Sample ~22 per category (110 / 5 = 22)
balanced = stratified_sample_by_category(unique_templates, n_per_category=22)

# Stratified split: 70/15/15
train, val, test = stratified_split(balanced, ratios=[0.7, 0.15, 0.15], stratify_by='category')
```

**Output Distribution:**
- Train: ~77 samples (15-16 per category)
- Val: ~16 samples (3-4 per category)
- Test: ~17 samples (3-4 per category)

**Why stratified?** Ensures each split has balanced category representation (critical for 5-class problem with limited data)

---

#### Step 3.3: Write Clean Data

```python
# Combine train/val/test with split labels
df['split'] = ...  # 'train', 'val', or 'test'
df.to_parquet('data/processed/clean_training_dev.parquet', index=False)
```

**Output:** `clean_training_dev.parquet` (~110 rows)
**Columns:** ticket_id, subject, description, category, error_code, template_id, split

**Schema Validation:** Pandera checks enforce:
- No NULL categories
- All splits present
- Balanced category distribution (±10%)

---

### [4] Traditional ML Training (Stage 3)

**Purpose:** Train and evaluate CatBoost and XGBoost on the clean split dataset.

**Core modules:**
- `src/models/catboost_model.py`
- `src/models/xgboost_model.py`
- `src/training/train_catboost.py`
- `src/training/train_xgboost.py`
- `src/training/run_training.py`
- `scripts/run_training.py`

**Training flow:**
1. Read `clean_training_<env>.parquet`
2. Fit `FeaturePipeline` for category classification
3. Build train/val/test matrices from the same feature contract
4. Optional Optuna search over bounded hyperparameter spaces
5. Train model, evaluate on test split, benchmark latency
6. Save model + per-class metrics + JSON summary + MLflow run

**Artifact contract:**
- Models: `models/catboost_category_<env>.cbm`, `models/xgboost_category_<env>.json`
- Summaries: `metrics/<env>/catboost_training_summary.json`, `metrics/<env>/xgboost_training_summary.json`
- Per-class metrics: `metrics/<env>/catboost_per_class.csv`, `metrics/<env>/xgboost_per_class.csv`

**Operational targets:**
- `make train-catboost`
- `make train-tradml`
- `poetry run python -m src.training.train_xgboost`

---

### [5] Retrieval System (Stage 5)

**Purpose:** Provide hybrid semantic + entity-aware ticket resolution lookup.

**Core modules:**
- `src/retrieval/corpus.py`
- `src/retrieval/embeddings.py`
- `src/retrieval/index.py`
- `src/retrieval/entities.py`
- `src/retrieval/search.py`
- `scripts/generate_embeddings.py`
- `scripts/build_search_index.py`

**Build flow:**
1. Load resolved tickets into `ResolutionDocument` objects
2. Embed source text with DistilBERT CLS vectors
3. Build FAISS `IndexFlatIP` over normalized vectors
4. Build entity inverted index for error code/product matches
5. Persist artifacts under `data/retrieval/`

**Runtime search flow:**
1. Embed incoming query text
2. Retrieve top-k semantic neighbors from FAISS
3. Boost scores using matched entities from query
4. Apply optional metadata filters (`category`, `product`, `resolution_code`)
5. Return enriched resolution payloads

**Artifacts:**
- `data/retrieval/faiss_index_<env>.bin`
- `data/retrieval/faiss_index_<env>.id_map.json`
- `data/retrieval/entity_index_<env>.json`
- `data/retrieval/corpus_<env>.json`

---

### [6] Sentiment + Anomaly (Stage 5)

**Purpose:** Add sentiment capability and monitoring primitives.

**Sentiment training:**
- `src/training/train_sentiment.py`
- Uses CatBoost only
- Input source: `mart_tickets_features` rows with non-empty `feedback_text`
- Output model: `models/catboost_sentiment_<env>.cbm`
- Output summary: `metrics/<env>/sentiment_training_summary.json`

**Anomaly modules:**
- `src/anomaly/detector.py` for per-prediction confidence anomalies
- `src/anomaly/volume_analyzer.py` for batch distribution/volume shifts
- `src/anomaly/baselines.py` for baseline creation and persistence
- `scripts/build_anomaly_baseline.py` to bootstrap baseline from clean training data

**Operational targets:**
- `make train-sentiment`
- `poetry run python scripts/build_search_index.py`
- `make build-anomaly-baseline`

---

## Technology Choices

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Storage** | DuckDB | Lightweight, embedded, SQL-compatible, no server overhead |
| **Transformations** | dbt | SQL-based, version-controlled, testable, auditable |
| **Splitting** | scikit-learn | Stratified sampling ensures balanced splits |
| **Data Validation** | Pandera | Schema enforcement, automated quality checks |
| **Orchestration** | Makefile | Simple, explicit, no complex DAG required for linear pipeline |
| **Configuration** | Python (config.py) | Type-safe, centralized, environment-aware |

---

## Design Decisions

Detailed decision rationale is maintained in [DECISIONS.md](DECISIONS.md). This section provides architectural context for decisions affecting system structure.

### Decision 1: Dual-Output Pipeline (Full + Clean)

See [D-006](DECISIONS.md#d-006-clean-training-data-strategy) and [D-007](DECISIONS.md#d-007-dual-output-pipeline) for full rationale.

**Context:** 100K dataset has 30% conflicting labels, but ~110 unique templates are clean

**Options Considered:**
1. **Train on full 100K** → 18% F1 (label noise drowns signal)
2. **Train on clean 110** → >85% F1 (deterministic mapping)
3. **Use full 100K for category, discard subcategory** → Still 30% conflicts

**Choice:** Option 2 (clean training set)

**Rationale:**
- Subject→category mapping is deterministic (~110 unique subjects)
- Clean data enables high-quality models (proven with BERT: 1.0 F1 on clean)
- Full 100K corpus still useful for RAG, search, anomaly detection
- Trade-off: Smaller training set, but clean labels → better models

**Evidence:** See `exploration/subcategory_independence/REPORT.md` for data quality analysis

---

### Decision 2: Separate Staging Layer (stg_tickets.sql)

**Context:** Type casting vs. business logic

**Options Considered:**
1. **Single mart layer** → Type casting mixed with feature extraction
2. **Separate staging + mart** → Clean separation of concerns

**Choice:** Option 2 (two-layer)

**Rationale:**
- Staging isolates type casting errors (easier to debug)
- Enables incremental testing (validate types before features)
- Follows dbt best practices (staging → marts → aggregates)

---

### Decision 3: Subject-Based Deduplication

**Context:** How to extract clean training set from 100K corpus

**Options Considered:**
1. **Random sampling** → Still includes conflicting labels
2. **Template_id-based** → Groups by full content hash
3. **Subject-based** → Groups by subject field only

**Choice:** Option 3 (subject-based)

**Rationale:**
- Investigation proved: subject→category mapping is deterministic
- Subject is the **primary signal** for category (description adds minimal value)
- Template_id groups are too granular (same subject + minor description variations)
- Subject provides ~110 clean templates (sufficient for 5-class problem)

**Evidence:** Subject→category mapping has zero conflicts in dataset (validated)

---

### Decision 4: DuckDB (Not PostgreSQL/MySQL)

**Context:** Which database for local development?

**Options Considered:**
1. **PostgreSQL** → Full-featured, but requires server process
2. **SQLite** → Lightweight, but limited SQL features
3. **DuckDB** → Analytical database, embedded, rich SQL

**Choice:** Option 3 (DuckDB)

**Rationale:**
- **Embedded:** No server process, single file database
- **Analytical:** Optimized for OLAP queries (group by, aggregations)
- **Rich SQL:** Window functions, CTEs, complex aggregations
- **Fast:** Columnar storage, vectorized execution
- **Parquet-native:** Direct read/write of Parquet files

**Trade-off:** Less mature than PostgreSQL, but sufficient for 100K dataset

---

### Decision 5: Stratified Splitting (Not Random)

**Context:** How to split ~110 samples into train/val/test?

**Options Considered:**
1. **Random split** → May result in imbalanced categories
2. **Stratified split** → Ensures balanced category distribution

**Choice:** Option 2 (stratified)

**Rationale:**
- 5 categories with ~22 samples each → small counts per class
- Random split could create val/test sets with missing categories
- Stratified sampling ensures each split has ~20% of each category
- Critical for 5-class problem with limited data

**Implementation:** `sklearn.model_selection.train_test_split(stratify=category)`

---

## Data Flow Metrics

| Stage | Input Rows | Output Rows | Quality Check |
|-------|-----------|-------------|---------------|
| Loader | 100,000 (JSON) | 100,000 (raw_tickets) | Schema validation |
| stg_tickets | 100,000 | 100,000 | Type casting checks |
| mart_tickets_features | 100,000 | 100,000 | Feature extraction validation |
| Splitter (dedup) | 100,000 | ~110 (unique subjects) | Conflict detection |
| Splitter (split) | ~110 | ~110 (with split labels) | Balance checks |
| **Final Output** | **100,000** | **~110** (clean_training_dev.parquet) | Pandera schema |

**Key Insight:** 100,000 → 110 reduction is **not data loss**, but deduplication to deterministic templates.

---

## Quality Assurance

### Automated Checks

1. **Schema Validation (Pandera)**
   - Column types match expected schema
   - No NULL in required fields (category, split)
   - Category values in allowed set

2. **Balance Validation**
   - Category distribution: ±10% of 20% (uniform 5-class)
   - Split distribution: train ≈ 70%, val ≈ 15%, test ≈ 15%

3. **Conflict Detection**
   - No subject with multiple categories
   - Deterministic subject→category mapping enforced

4. **dbt Tests**
   - Row count checks (staging = source count)
   - NOT NULL constraints on key fields
   - Unique constraints on ticket_id

### Manual Inspection

1. **Sample Review**
   - Print first 10 rows of each split
   - Verify subjects look reasonable
   - Check category distribution

2. **Conflict Report**
   - List any subjects with multiple categories (should be empty)
   - Flag for manual review if conflicts detected

---

## Environment Configurations

### Smoke Test (ENV=dev SMOKE_TEST=true)
- Input: First 1000 rows from support_tickets.json
- Expected clean templates: ~10-20
- Runtime: ~10 seconds
- Purpose: Fast validation during development

### Development (ENV=dev)
- Input: Full 100K rows
- Expected clean templates: ~110
- Runtime: ~1-2 minutes
- Purpose: Full pipeline testing locally

### Production (ENV=prod)
- Input: Full 100K rows
- DuckDB: Persistent database file
- Output: Validated parquet files
- Purpose: Final data preparation for model training

---

## Extensibility

### Adding New Features (dbt)

```sql
-- In mart_tickets_features.sql
SELECT
    *,
    CASE
        WHEN description LIKE '%timeout%' THEN 'timeout_error'
        WHEN description LIKE '%connection%' THEN 'connection_error'
        ELSE 'other'
    END AS error_category
FROM {{ ref('stg_tickets') }}
```

**Benefit:** SQL-based transformations are auditable and version-controlled

---

### Changing Split Ratios

```python
# In src/data/splitter.py
train, temp = train_test_split(df, train_size=0.8, stratify=df['category'])
val, test = train_test_split(temp, train_size=0.5, stratify=temp['category'])
# New split: 80/10/10
```

**Benefit:** Centralized configuration, easy to adjust for experiments

---

## Limitations & Future Work

### Current Limitations

1. **Small training set** (~110 samples)
   - Sufficient for high-signal 5-class problem
   - May not scale to 25-class (subcategory) or 100-class problems
   - Mitigated by: Clean data with deterministic mapping

2. **No incremental updates**
   - Full pipeline re-runs on each execution
   - Acceptable for 100K dataset (fast)
   - May need incremental dbt models for larger datasets

3. **Single data source**
   - Only support_tickets.json
   - No joins with external data (customer info, product metadata)
   - Future: Add dimensional tables for enrichment

### Future Enhancements

1. **Data Augmentation**
   - Generate synthetic variants of clean templates
   - Paraphrase subjects/descriptions (LLM-based)
   - Goal: Expand training set while maintaining label quality

2. **Active Learning Pipeline**
   - Model predicts on full 100K corpus
   - Flag low-confidence predictions for human review
   - Add reviewed samples to clean training set
   - Iterative improvement loop

3. **Feature Engineering**
   - TF-IDF vectors from subject/description
   - Error code co-occurrence patterns
   - Temporal features (hour of day, day of week)

---

## Success Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Clean templates extracted | ~110 | ~110 | ✓ |
| Subject→category conflicts | 0 | 0 | ✓ |
| Category balance (train) | 18-22% per class | 19-21% | ✓ |
| Split ratios | 70/15/15 ±2% | 70/14.5/15.5 | ✓ |
| Pipeline runtime (full) | <5 min | ~2 min | ✓ |
| Pipeline runtime (smoke) | <30 sec | ~10 sec | ✓ |

---

## References

- **Data Quality Investigation:** `exploration/subcategory_independence/REPORT.md`
- **dbt Models:** `dbt_project/models/`
- **Splitter Implementation:** `src/data/splitter.py`
- **Configuration:** `src/config.py`
- **Orchestration:** `Makefile` and Python scripts (`make data-pipeline`, `make train-tradml`, `make train-bert`)

---

### [7] API Layer (Stage 6)

**Purpose:** Serve predictions, sentiment analysis, and search via FastAPI REST endpoints.

**Core modules:**
- `src/api/app.py` - FastAPI application with /predict, /analyze-feedback, /health
- `src/api/schemas.py` - Pydantic request/response models
- `src/api/search.py` - /search endpoint router

**Endpoints:**

1. **POST /predict**
   - Input: TicketInputMultiClassifier (subject, description, metadata)
   - Output: predicted category (real), priority + sentiment (placeholders)
   - Backend: configurable (CatBoost, XGBoost, or BERT via `SERVING_BACKEND`)
   - Warning flags: `priority_placeholder`, `sentiment_placeholder`, `low_confidence`, `confidence_anomaly`, `possible_leakage_pattern`

2. **POST /analyze-feedback**
   - Input: ticket_id + feedback_text
   - Output: real sentiment prediction (CatBoost)
   - Used for post-resolution customer feedback analysis

3. **POST /search**
   - Input: query + optional filters (category, product, resolution_code)
   - Output: top-k similar historical resolutions
   - Backend: SearchEngine (FAISS + entity matching)

4. **GET /health**
   - Returns: backend readiness, anomaly detector status, search index status

**ModelManager:**
- Loads category models (CatBoost, XGBoost, BERT) at startup
- Loads sentiment model (CatBoost)
- Loads feature pipelines
- Loads anomaly detector baseline
- Validates backend readiness before serving requests

**Prediction flow:**
1. Validate request schema (Pydantic)
2. Convert to DataFrame
3. Check for leakage patterns in text
4. Transform via FeaturePipeline (TF-IDF + tabular features)
5. Predict with backend model
6. Normalize probability distribution
7. Check confidence threshold
8. Check anomaly detector (if loaded)
9. Return response with warnings

**Artifacts:**
- Contract version: 1.0.0
- Default backend: XGBoost
- Confidence threshold: 0.5
- Serving port: 8000

**Operational targets:**
- `make api` - start dev server with auto-reload
- `make docker-build` - build Docker image
- `make docker-up` - start containerized API
