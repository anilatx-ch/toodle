# Technical Decisions Log

**Purpose:** Decision log with rationale for architectural and technical choices
**Status:** Living document - updated each stage
**Current Stage:** Stage 5 (Sentiment, Search & Anomaly) - Complete
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

### Stage 2: Feature Engineering (COMPLETE)

#### D-008: Feature Engineering Approach

**Decision:** TF-IDF for text + one-hot encoding for categoricals + standard scaling for numericals

**Rationale:**
- Simple, interpretable features work well with gradient boosting (CatBoost/XGBoost)
- TF-IDF captures term importance without embedding overhead (max_features=10000, chi2 k=5000)
- Categorical one-hot encoding preserves interpretability for feature importance
- Standard scaling ensures numeric features on comparable scales
- Less semantic richness than embeddings, but faster and more debuggable
- BERT path available for deep learning comparison (Stage 4)

**Trade-offs:**
- TF-IDF doesn't capture word order or context (acceptable for clean 5-class problem)
- Fixed vocabulary means unseen words at inference are ignored (mitigated by large vocab)

**Evidence:** Original DOODLE achieved 88% F1 with TF-IDF features on clean data

**Status:** Implemented

---

#### D-009: Feature Pipeline Simplification

**Decision:** Single unified pipeline per classifier (remove multi-classifier factory)

**Rationale:**
- Original design anticipated batch processing multiple targets (category, priority, sentiment) in parallel
- With only category as real target in TOODLE (priority/sentiment are placeholders), factory adds complexity without benefit
- `create_for_classifier(name)` provides explicit classifier selection when needed
- Simpler to reason about single pipeline lifecycle: create → fit → transform → save
- YAGNI principle - can add factory back if multi-classifier batch becomes requirement

**Trade-offs:**
- Less flexibility for future multi-classifier batch processing
- If priority/sentiment become real, callers create pipelines separately
- But: Cleaner code now, easier to understand and maintain

**Status:** Implemented

---

### Stage 2.5: Evaluation & Experiment Tracking

#### D-010: Evaluation Metrics

**Decision:** Weighted F1 as primary metric, with per-class precision/recall

**Rationale:** Balanced dataset (~22 per class) makes weighted F1 appropriate. Per-class metrics help identify category-specific issues. Weighted F1 accounts for any minor class imbalances while remaining interpretable.

**Trade-off:** Standard metrics; no custom business-weighted scoring. Simpler approach appropriate for assessment scope.

**Status:** Implemented

---

#### D-011: MLflow Experiment Structure

**Decision:** Single experiment per model type, runs tagged by data version and config hash

**Rationale:** Simple structure for assessment scope. Tags enable filtering without complex experiment hierarchy. Avoids over-engineering while maintaining experiment traceability.

**Trade-off:** Less organization than production MLOps, appropriate for demo. In production, would use nested experiments (model type → hyperparameter search → final model).

**Status:** Implemented

---

### Stage 3: Traditional ML Models

#### D-013: Hyperparameter Tuning Strategy

**Decision:** Use Optuna with a small, bounded search budget (10 trials in smoke mode, 20 trials in full mode).

**Rationale:** The clean category dataset is intentionally small (~110 balanced samples). Large search budgets overfit quickly and inflate runtime for minimal gain. A constrained search over core CatBoost/XGBoost knobs provides enough exploration while preserving generalization and reproducibility.

**Trade-offs:** Lower chance of finding a globally optimal configuration vs substantially lower overfitting risk and faster iteration.

**Status:** Implemented

### Stage 4: Deep Learning (BERT)

#### D-014: BERT Training Configuration

**Decision:** Use small batch (16), few epochs (4), and early stopping (patience=2) for DistilBERT.

**Rationale:** With ~110 clean training samples, BERT can overfit quickly. This setup keeps training stable while limiting unnecessary epochs.

**Trade-off:** This conservative schedule may underfit harder variants, but is appropriate for the deterministic clean-data category task.

**Status:** Implemented

---

#### D-015: Text-Only as Primary BERT Path

**Decision:** Make text-only BERT training the default path, with optional tabular fusion retained in the model wrapper.

**Rationale:** Category prediction signal is concentrated in ticket subject/description. Text-only mode reduces pipeline dependencies while preserving multimodal capability for future extensions.

**Trade-off:** Tabular features are not used in default Stage 4 training, but optional model support remains available to avoid redesign later.

**Status:** Implemented

### Stage 5: Retrieval, Sentiment & Anomaly Detection

#### D-016: Retrieval Strategy

**Decision:** Use hybrid retrieval with FAISS semantic search + entity keyword boosting.

**Rationale:** Semantic vectors capture intent similarity across ticket phrasing, while entity matching (error codes, product names) improves precision for operational lookups. The combined strategy demonstrates both dense retrieval and structured signal usage in a compact design.

**Trade-off:** Requires index build artifacts (vector index + entity index + corpus snapshot) and introduces an embedding dependency for index generation.

**Status:** Implemented

---

#### D-017: Anomaly Detection Scope

**Decision:** Implement confidence-based per-category anomaly detection and batch volume-distribution shift analysis (JSD + volume delta).

**Rationale:** These two signals cover common failure modes for ticket systems:
- low-confidence predictions inside a known category
- sudden category distribution shifts indicating emerging incidents

The approach stays interpretable and lightweight for assessment scope.

**Trade-off:** This does not detect subtle semantic drift directly; it focuses on confidence and volume behavior.

**Status:** Implemented

---

#### D-018: Sentiment Backend Simplification

**Decision:** Use CatBoost only for sentiment classification.

**Rationale:** Sentiment is a secondary classifier in this project. A single traditional backend demonstrates capability without duplicating training code and model artifacts across multiple frameworks.

**Trade-off:** No backend comparison for sentiment specifically, but lower maintenance surface and faster iteration.

**Status:** Implemented

### Stage 6: API Layer
Decisions for API design, deployment strategy will be added here.

---

## Cross-References

- **System Architecture**: See `ARCHITECTURE.md` for implementation details
- **Model Performance**: See `MODEL.md` for benchmarks and analysis
- **Data Investigation**: See `exploration/subcategory_independence/REPORT.md` for statistical evidence
- **Porting Plan**: See `PLAN_PORTING.md` for stage-by-stage roadmap
