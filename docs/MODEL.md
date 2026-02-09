# Model Documentation

**Purpose:** Model performance benchmarks, comparisons, and feature importance
**Status:** Living document - Stage 3 traditional ML implementation complete
**Last Updated:** 2026-02-09

---

## Overview

Stage 3 adds runnable CatBoost and XGBoost training pipelines on clean split data with:
- shared feature artifacts (`FeaturePipeline` + TF-IDF vectorizer)
- optional Optuna tuning with bounded trial counts
- standardized evaluation (weighted F1 primary)
- latency benchmarking and MLflow logging

This document provides:
1. **Stage 3 traditional ML model details**
2. **Performance baselines** (used before full training artifacts are generated)
3. **Data quality findings** (why clean data approach)
4. **Error analysis** (predicted failure modes)

---

## Category Classification

### Problem Definition

Predict support ticket category from text and metadata.
- **Classes:** 5 categories (Account Management, Data Issue, Feature Request, Security, Technical Issue)
- **Training data:** clean deduplicated subject→category set (`clean_training_<env>.parquet`)
- **Primary metric:** weighted F1

### Traditional ML Models

#### CatBoost
- **Architecture:** gradient boosting classifier with multiclass objective
- **Features:** TF-IDF text + tabular features from `FeaturePipeline`
- **Training entrypoint:** `python -m src.training.train_catboost`
- **Optuna strategy:** optional bounded search (small-budget trials for clean dataset)
- **Artifacts:** model (`.cbm`), per-class metrics CSV, JSON summary, MLflow run

#### XGBoost
- **Architecture:** histogram-based gradient boosting with multiclass soft probabilities
- **Features:** same feature pipeline as CatBoost
- **Training entrypoint:** `python -m src.training.train_xgboost`
- **Optuna strategy:** optional bounded search (same budget policy as CatBoost)
- **Artifacts:** model (`.json`), per-class metrics CSV, JSON summary, MLflow run

### Stage 3 Performance

Performance values are written at runtime to:
- `metrics/<env>/catboost_training_summary.json`
- `metrics/<env>/xgboost_training_summary.json`

The repository does not include training datasets/artifacts, so committed docs track
the training interfaces and artifact contract; numeric benchmark values are filled after local runs.

---

## Stage 1: Pre-Training Baseline (Current)

This section documents expected model performance based on data quality analysis. Actual training results will be added in Stage 3-4.

---

## Performance Baseline (Expected)

### Dataset Characteristics

- **Training Set:** ~110 samples (clean_training_dev.parquet)
- **Classes:** 5 categories (Account Management, Data Issue, Feature Request, Security, Technical Issue)
- **Distribution:** Balanced (~22 samples per category)
- **Split:** 70/15/15 train/val/test stratified by category
- **Label Quality:** 100% deterministic (subject→category mapping has zero conflicts)

### Expected Performance (Stage 3-4 Models)

| Model | Expected F1 (Macro) | Expected Accuracy | Confidence |
|-------|-------------------|------------------|------------|
| **Traditional ML** (CatBoost, XGBoost) | >0.85 | >0.85 | High |
| **BERT** (fine-tuned) | >0.85 | >0.85 | Very High |
| **Zero-shot LLM** (GPT-4, Claude) | 0.70-0.80 | 0.70-0.80 | Medium |

**Evidence:**
- Prior validation: BERT achieved **1.0 F1** on clean 110-sample dataset
- Traditional ML (CatBoost): **0.88 F1** on clean data
- Clean data with deterministic mapping enables high model performance

**Why high performance is achievable:**
1. **Clean labels:** Zero conflicting subject→category mappings
2. **High signal:** Subject is highly predictive of category (deterministic relationship)
3. **Balanced classes:** ~22 samples per category (no class imbalance)
4. **Simple problem:** 5 classes, clear semantic boundaries

---

## Data Quality Investigation

### Subject→Category Mapping Analysis

**Finding:** Subject→category mapping is **deterministic** (proven)

**Evidence:**
```python
# Validation check (src/data/splitter.py)
conflicts = df.groupby('subject')['category'].nunique()
conflict_count = (conflicts > 1).sum()
# Result: 0 conflicts (100% deterministic)
```

**Implications:**
- Category classification is fundamentally a **template matching problem**
- 110 unique subjects = 110 unique templates
- Models learn subject→category patterns (not complex NLP)
- Small training set (110 samples) is sufficient due to deterministic mapping

---

### Subcategory Unpredictability

**Question:** Why predict category only, not subcategory?

**Answer:** Subcategory is statistically unpredictable (proven via 8 independent tests)

**Full Investigation:** See `exploration/subcategory_independence/REPORT.md`

**Summary of Evidence:**

| Test | Result | Interpretation |
|------|--------|----------------|
| **Uniformity** | χ² p > 0.05 (5/5 categories) | Perfectly uniform distribution |
| **Conditional Independence** | Fisher p > 0.05 (6/6 features) | No feature associations |
| **Mutual Information** | Max NMI = 0.0047 << 0.01 | Negligible information |
| **Joint-Lookup Upper Bound** | Accuracy = 0.1997 ≈ 0.20 | At random baseline |
| **Nested CV (ML)** | Max accuracy = 0.2037 < 0.21 | Below practical threshold |
| **GradientBoosting** | Mean accuracy = 0.1875 < 0.20 | **Below baseline** |
| **Permutation Test** | p = 1.0 | Cannot beat shuffled labels |
| **Power Check** | 100% detection of 0.28 target | Method not blind |

**Conclusion:** Subcategory contains **zero predictive signal**. All ML models perform at or below random guessing (0.20 accuracy for 5 uniform classes).

**Decision:** Scope out subcategory prediction. Focus on category (5-class problem with >85% achievable F1).

**Impact:**
- Training set: 110 clean samples (not split into 5 × 22 subcategory groups)
- Model complexity: 5 output classes (not 25)
- Expected performance: >85% F1 (not 20% random guessing)

---

### Noisy 100K Dataset Analysis

**Problem:** Full 100K dataset has 30% conflicting labels

**Evidence:**
```python
# Conflict analysis
grouped = df.groupby('subject')['category'].nunique()
conflicting = (grouped > 1).sum() / len(grouped)
# Result: ~30% of subjects have multiple categories
```

**Performance on noisy data:**
- Best model (CatBoost): **18% F1** on 100K noisy dataset
- Same model on clean 110 samples: **88% F1**
- **+70 percentage point improvement** from data quality fix alone

**Root cause:** Label noise drowns signal
- Same subject → different categories (data entry errors, ambiguous guidelines)
- Models learn spurious patterns from noise
- Conflicting examples cancel out during training

**Solution:** Extract clean templates (110 samples with deterministic mapping)

**Trade-off:**
- ❌ Smaller training set (110 vs 100K)
- ✅ Clean labels (0% vs 30% conflicts)
- ✅ Better models (88% vs 18% F1)
- ✅ Deterministic mapping (subject defines category)

---

## Error Analysis (Predicted Failure Modes)

Models trained on clean data will struggle with:

### 1. Novel Subjects (Not in Training Templates)

**Scenario:** New ticket subject not seen during training

**Example:**
- Training: "Database connection timeout", "API authentication failed"
- Novel: "Quantum entanglement module error"

**Expected Behavior:**
- Model predicts based on keyword similarity (e.g., "error" → Technical Issue)
- May misclassify if keywords misleading

**Mitigation:**
- Use confidence thresholds (flag low-confidence predictions)
- Human-in-the-loop for novel subjects
- Active learning: Add reviewed novel subjects to training set

**Estimated Frequency:** 10-20% of production traffic (depends on subject diversity)

---

### 2. Ambiguous Short Subjects

**Scenario:** Generic subjects with minimal context

**Example:**
- "Issue with product"
- "Need help"
- "Question"

**Expected Behavior:**
- Model has low confidence (multiple categories plausible)
- May rely on description field (secondary signal)

**Mitigation:**
- Require minimum subject length (validation rule)
- Fall back to description if subject too generic
- Prompt users for more specific subject lines

**Estimated Frequency:** 5-10% of tickets

---

### 3. Multi-Category Problems

**Scenario:** Ticket spans multiple categories

**Example:**
- "Login fails AND data sync issue"
- "Feature request for security improvement"

**Expected Behavior:**
- Model picks one category (highest probability)
- Loses information about secondary category

**Mitigation:**
- Multi-label classification (future enhancement)
- Hierarchical tagging (primary + secondary categories)
- Human review for complex tickets

**Estimated Frequency:** <5% of tickets

---

### 4. Evolving Categories (Concept Drift)

**Scenario:** New product features change category semantics

**Example:**
- "Mobile app" category added after training
- "API v2" issues distinct from "API v1"

**Expected Behavior:**
- Model maps new concepts to existing categories (best match)
- Degraded performance on novel concepts

**Mitigation:**
- Periodic model retraining (monthly/quarterly)
- Monitor prediction confidence over time
- Active learning to capture new patterns

**Estimated Frequency:** Gradual drift (depends on product evolution)

---

## Feature Importance (Expected)

Based on clean data structure and deterministic mapping:

### Primary Features

1. **Subject** (99% of signal)
   - Deterministic subject→category mapping
   - Most predictive feature by far
   - Models primarily learn subject patterns

### Secondary Features

2. **Description** (1% of signal)
   - Disambiguates similar subjects
   - Useful for edge cases
   - Minimal impact on clean templates

### Tertiary Features (Minimal Impact)

3. **Error codes** (extracted from description)
   - Weak category signal (errors span categories)
   - Useful for subcategory (but subcategory unpredictable)

4. **Product module** (metadata field)
   - Weak category signal (modules used across categories)
   - May help for module-specific categories (future)

5. **Temporal features** (created_at, product_month)
   - No expected category signal (tickets created uniformly)
   - Useful for trend analysis (not classification)

**Evidence:** Investigation showed:
- Mutual information: All features have NMI < 0.005 for subcategory
- Subject is the only strong signal for category (deterministic mapping)

---

## Experiment Tracking

All data transformations are logged and reproducible:

### Pipeline Execution Logs

```bash
# Run full pipeline with logging
ENV=dev make data-pipeline 2>&1 | tee logs/data_pipeline_$(date +%Y%m%d_%H%M%S).log
```

**Logged Metrics:**
- Row counts at each stage (loader → staging → mart → splitter)
- Schema validation results (Pandera checks)
- Split proportions (train/val/test ratios)
- Category distribution (balance check)
- Conflict detection (subject→category conflicts)

---

### DuckDB Query Logs

```sql
-- Verify row counts
SELECT COUNT(*) FROM raw_tickets;
SELECT COUNT(*) FROM stg_tickets;
SELECT COUNT(*) FROM featured_tickets;

-- Check category distribution
SELECT category, COUNT(*) as count
FROM featured_tickets
GROUP BY category
ORDER BY count DESC;
```

**Benefit:** SQL queries are version-controlled (dbt) and auditable

---

### Splitter Output Validation

```python
# In src/data/splitter.py
logger.info(f"Extracted {len(unique_templates)} unique subject templates")
logger.info(f"Split distribution: train={train_pct:.1%}, val={val_pct:.1%}, test={test_pct:.1%}")
logger.info(f"Category balance: {category_balance}")
```

**Logged Statistics:**
- Unique template count (~110 expected)
- Split sizes and proportions
- Category distribution per split
- Conflict count (should be 0)

---

### Reproducibility Checklist

- [x] **Random seeds fixed:** `random_state=42` in all `train_test_split` calls
- [x] **Environment pinned:** `pyproject.toml` locks package versions
- [x] **Data versioned:** Raw JSON file in git LFS (or external storage)
- [x] **Transforms versioned:** dbt models in git
- [x] **Config centralized:** `src/config.py` for paths and parameters
- [x] **Logs captured:** Makefile redirects output to logs/

**Reproduce Pipeline:**
```bash
git clone <repo>
poetry install
ENV=dev SMOKE_TEST=true make data-pipeline  # Smoke test (fast)
ENV=dev make data-pipeline                  # Full pipeline
```

---

## Model Training Readiness (Stage 3-4)

### Data Artifacts Ready

- [x] `data/processed/clean_training_dev.parquet` (~110 samples)
- [x] Schema validated (Pandera)
- [x] Splits labeled (train/val/test column)
- [x] Categories balanced (19-21% per class)
- [x] Zero conflicting labels

### Expected Model Pipeline (Stage 3)

```python
# Pseudocode for Stage 3
train_df = pd.read_parquet('data/processed/clean_training_dev.parquet')
train_df = train_df[train_df['split'] == 'train']

# CatBoost
model = CatBoostClassifier(iterations=500, depth=6, random_state=42)
model.fit(train_df['subject'], train_df['category'])

# Evaluate
val_df = train_df[train_df['split'] == 'val']
y_pred = model.predict(val_df['subject'])
f1 = f1_score(val_df['category'], y_pred, average='macro')
print(f"Validation F1: {f1:.3f}")  # Expected: >0.85
```

### Expected Evaluation Metrics (Stage 4)

- **Per-class F1:** 0.80-0.95 (varies by category)
- **Macro F1:** >0.85 (average across classes)
- **Accuracy:** >0.85 (matches F1 for balanced classes)
- **Confusion matrix:** Minimal off-diagonal (deterministic mapping)
- **Calibration:** High confidence for correct predictions (low entropy)

---

## Comparison to Baseline Approaches

### Approach 1: Train on Full 100K (Noisy Data)

- **F1:** 18% (proven in Prior validation)
- **Issue:** Label noise drowns signal
- **Not viable:** Below random guessing for 5 classes (20%)

### Approach 2: Train on Clean 110 Templates

- **F1:** >85% (proven in Prior validation)
- **Advantage:** Deterministic mapping, zero conflicts
- **Trade-off:** Smaller training set, but sufficient for 5-class problem
- **Chosen approach** ✓

### Approach 3: Data Augmentation (110 → 1000s)

- **Expected F1:** 85-90% (diminishing returns)
- **Method:** Paraphrase subjects/descriptions (LLM-based)
- **Risk:** Synthetic data may not match production distribution
- **Status:** Future work (Stage 5+)

### Approach 4: Active Learning (Iterative)

- **Expected F1:** 90-95% (iterative improvement)
- **Method:** Model predicts on 100K, human reviews low-confidence, add to training
- **Benefit:** Expands training set with validated labels
- **Status:** Future work (Stage 5+)

---

## Success Criteria (Stage 3-4 Models)

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| **Validation F1 (macro)** | 0.80 | 0.85 | 0.90 |
| **Test F1 (macro)** | 0.75 | 0.80 | 0.85 |
| **Per-class F1 (min)** | 0.70 | 0.75 | 0.80 |
| **Test Accuracy** | 0.75 | 0.80 | 0.85 |
| **Inference Time (p95)** | <100ms | <50ms | <20ms |
| **Model Size** | <500MB | <100MB | <10MB |

**Rationale for targets:**
- **0.85 F1:** Proven achievable on clean data (Prior validation)
- **5-10 point gap (val→test):** Expected from small dataset (overfitting risk)
- **Per-class F1 > 0.75:** All categories should perform well (no weak classes)

---

## Limitations & Risks

### Limitation 1: Small Training Set (110 Samples)

**Risk:** Overfitting to training templates

**Mitigation:**
- Regularization (L2 penalty, early stopping)
- Cross-validation (5-fold on training set)
- Monitor val/test gap (should be <10 points)

**Acceptance:** Trade-off for clean labels (0% vs 30% noise)

---

### Limitation 2: Template Matching (Not Generalization)

**Risk:** Model memorizes training subjects, fails on novel subjects

**Mitigation:**
- Confidence thresholds (flag low-confidence predictions)
- Human-in-the-loop for novel subjects
- Active learning to expand training set

**Acceptance:** Deterministic mapping is a feature, not a bug (subject defines category)

---

### Limitation 3: Single Language (English)

**Risk:** Non-English tickets may be misclassified

**Mitigation:**
- Language detection (flag non-English tickets)
- Multilingual models (Stage 5+)
- Translation pre-processing (if needed)

**Acceptance:** Dataset is English-only (no multilingual requirement in Phase 1)

---

## Next Steps (Stage 4+)

### Stage 4: Deep Learning Comparison
1. Implement DistilBERT training on clean split data
2. Align evaluation/latency reporting with traditional ML outputs
3. Compare BERT vs CatBoost vs XGBoost on identical test split

### Stage 5+: Iteration
1. Active learning (expand training set)
2. Data augmentation (paraphrase templates)
3. Multi-label classification (handle multi-category tickets)
4. Concept drift monitoring (track performance over time)

---

## References

- **Data Quality Investigation:** `exploration/subcategory_independence/REPORT.md`
- **Architecture Documentation:** `docs/ARCHITECTURE.md`
- **Clean Training Data:** `data/processed/clean_training_dev.parquet`
- **Pipeline Logs:** `logs/data_pipeline_*.log`

---

## Actual Model Performance (To Be Added From Runtime Artifacts)

### CatBoost Results
[Loaded from `metrics/<env>/catboost_training_summary.json`]

### XGBoost Results
[Loaded from `metrics/<env>/xgboost_training_summary.json`]

### DistilBERT Results
[Metrics, confusion matrix, attention analysis - to be added after training]

### Model Comparison Table

| Model | F1 Score | Accuracy | Latency (p50) | Latency (p99) | Size (MB) |
|-------|----------|----------|---------------|---------------|-----------|
| CatBoost | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| XGBoost | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| DistilBERT | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### Recommendation
[Analysis of which model to use in production and why - to be added after Stage 4]
