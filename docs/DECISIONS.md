# Technical Decisions Log

**Purpose:** Decision log with rationale for architectural and technical choices
**Status:** Living document - updated each stage
**Current Stage:** Stage 1 (Data Pipeline) - Complete
**Last Updated:** 2026-02-09

---

## Format

Each decision entry follows this structure:

- **Decision:** What was chosen
- **Rationale:** Why this choice was made
- **Trade-offs:** Costs and benefits of the decision
- **Evidence:** Supporting data or references (where applicable)
- **Status:** Implemented | Planned | Deferred

---

## Stage 0 Decisions: Scaffold & Config

### D-001: Traditional ML Framework Selection

**Decision:** CatBoost + XGBoost (drop LightGBM)

**Rationale:** Two gradient boosting implementations suffice to demonstrate comparison skills. LightGBM adds marginal value while increasing maintenance surface. CatBoost offers native categorical handling; XGBoost is industry standard.

**Trade-offs:** Less framework breadth vs cleaner codebase.

**Status:** Implemented

---

### D-002: Deep Learning Framework

**Decision:** DistilBERT via keras-nlp on TensorFlow

**Rationale:** Assessment specifies "TensorFlow/Keras". DistilBERT balances accuracy with inference speed for a demo system.

**Trade-offs:** Heavier dependency than sklearn, but demonstrates transformer competency.

**Status:** Implemented

---

### D-003: Data Pipeline Stack

**Decision:** dbt + DuckDB (in-process)

**Rationale:** Job spec values dbt experience. DuckDB avoids external database dependency while supporting SQL transforms. Demonstrates data engineering without infrastructure overhead.

**Trade-offs:** Not production-scale, but appropriate for assessment scope.

**Status:** Implemented

---

### D-004: Experiment Tracking

**Decision:** MLflow local tracking

**Rationale:** Job spec explicitly lists MLflow. Local file store avoids server dependency.

**Trade-offs:** No remote collaboration features, acceptable for single-developer demo.

**Status:** Implemented

---

### D-005: Environment Simplification

**Decision:** SMOKE_TEST boolean + single ENV variable

**Rationale:** Original 3-tier (dev/test/prod) added complexity without value in assessment context. SMOKE_TEST controls data size; ENV controls path namespacing.

**Trade-offs:** Less production-realistic, but cleaner for demo purposes.

**Status:** Implemented

---

## Stage 1 Decisions: Data Pipeline

### D-006: Clean Training Data Strategy

**Decision:** Train all models on ~110 deduplicated subject→category pairs, not noisy 100K

**Context:** Investigation revealed subject→category mapping is deterministic (~110 unique subjects with zero conflicts). The 100K dataset contains ~30% label noise from the generation process.

**Rationale:** Training on clean data is the correct approach; noisy training was a data quality bug, not a feature. The subject field alone provides deterministic category mapping, proven through validation checks that found zero conflicting labels.

**Evidence:**
- BERT prototype on clean data: **1.0 F1**
- CatBoost baseline on clean data: **0.88 F1** vs **0.18 F1** on noisy 100K
- Subject→category validation: **0 conflicts** out of ~110 unique subjects
- See `exploration/subcategory_independence/REPORT.md` for full statistical analysis
- See `MODEL.md` for performance comparison details

**Trade-offs:**
- Smaller training set (110 vs 100K samples)
- Clean labels (0% vs 30% conflicts)
- Expected model performance: >85% F1 vs 18% on noisy data

**Status:** Implemented

---

### D-007: Dual Output Pipeline

**Decision:** dbt processes full 100K (for RAG corpus), splitter extracts clean ~110 (for training)

**Rationale:** Full corpus needed for semantic search and anomaly baselines. Clean subset needed for accurate classification training. Both derive from same source with different purposes.

**Architecture:**
1. **Full pipeline path**: support_tickets.json → loader → dbt (stg_tickets, mart_tickets_features) → 100K featured_tickets
   - **Purpose**: RAG corpus, semantic search, embeddings, anomaly detection baselines
   - **Characteristics**: Preserves all tickets, includes label noise, rich features

2. **Clean training path**: featured_tickets → splitter (deduplicate, balance, stratify) → ~110 clean_training_dev.parquet
   - **Purpose**: Model training (CatBoost, XGBoost, BERT)
   - **Characteristics**: Deterministic labels, balanced classes, stratified splits

**Trade-offs:** Two data paths to maintain, but each serves distinct purpose. Added complexity justified by serving different use cases (retrieval vs classification).

**Evidence:** See `ARCHITECTURE.md` for detailed data flow and component design.

**Status:** Implemented

---

## Future Stages (Planned)

### Stage 2: Feature Engineering
Decisions for TF-IDF, embeddings, and feature selection will be added here.

### Stage 3: Traditional ML Models
Decisions for CatBoost and XGBoost hyperparameters, training strategy will be added here.

### Stage 4: Deep Learning (BERT)
Decisions for fine-tuning approach, architecture choices will be added here.

### Stage 5: Retrieval & Anomaly Detection
Decisions for FAISS configuration, anomaly thresholds will be added here.

### Stage 6: API Layer
Decisions for API design, deployment strategy will be added here.

---

## Cross-References

- **System Architecture**: See `ARCHITECTURE.md` for implementation details
- **Model Performance**: See `MODEL.md` for benchmarks and analysis
- **Data Investigation**: See `exploration/subcategory_independence/REPORT.md` for statistical evidence
- **Porting Plan**: See `PLAN_PORTING.md` for stage-by-stage roadmap
