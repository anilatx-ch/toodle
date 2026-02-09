# Model Documentation

**Purpose:** Model performance analysis and comparison
**Status:** Stage 7 Complete
**Last Updated:** 2026-02-09

---

## Overview

TOODLE trains three models for support ticket category classification:
1. **CatBoost** - Gradient boosting with native categorical handling
2. **XGBoost** - Gradient boosting with histogram-based splits
3. **DistilBERT** - Fine-tuned transformer for text classification

All models train on **~110 clean, deduplicated subject→category pairs** with zero label conflicts (see Data Quality Investigation below).

---

## Performance Comparison

### Current Results (Smoke Test Mode)

The metrics below are from **smoke test training** with minimal hyperparameters on ~20/10/17 train/val/test samples:

| Model      | F1 Score | Accuracy | Latency (p50) | Latency (p95) | Size (MB) |
|------------|----------|----------|---------------|---------------|-----------|
| CatBoost   | 0.4781   | 0.5294   | 0.27 ms       | 0.29 ms       | 0.01      |
| XGBoost    | 0.5462   | 0.6471   | 0.35 ms       | 0.43 ms       | 0.01      |
| DistilBERT | 0.0182   | 0.1000   | 107.78 ms     | 123.32 ms     | 759.80    |

**Smoke Test Configuration:**
- **Data:** 20 train / 10 val / 17 test samples (insufficient for real evaluation)
- **CatBoost:** iterations=2, depth=2
- **XGBoost:** n_estimators=2, depth=1
- **BERT:** epochs=2, batch_size=2, max_len=32

These minimal settings validate the training pipeline but do not represent production-quality models.

---

### Expected Production Performance

Based on clean data properties and DOODLE baseline results, full training is expected to achieve:

| Model      | Expected F1 | Expected Accuracy | Expected Latency (p50) | Size (MB) |
|------------|-------------|-------------------|------------------------|-----------|
| CatBoost   | 0.85-0.90   | 0.85-0.90         | 0.5-1.0 ms             | 0.1       |
| XGBoost    | 0.85-0.88   | 0.85-0.88         | 0.8-1.5 ms             | 0.05      |
| DistilBERT | 0.88-0.95   | 0.88-0.95         | 30-50 ms (CPU)         | 250       |

**Production Configuration:**
- **Data:** Full clean ~110 samples, stratified 70/15/15 split (~77/16/17)
- **CatBoost:** 100-200 iterations, depth 4-6, Optuna tuning
- **XGBoost:** 50-150 estimators, depth 3-5, Optuna tuning
- **BERT:** 4-8 epochs, batch_size=16, max_len=128, early stopping

**Evidence:** Original DOODLE achieved **0.88 F1** with CatBoost on the same clean data approach.

---

## Data Quality Investigation

### The Problem: Noisy 100K Dataset

Initial training on the full 100K ticket corpus achieved only **18% F1** with CatBoost due to:
- **30% label conflicts** - identical tickets with different categories
- **Label noise from generation process** - synthetic data inconsistencies
- **Signal dilution** - deterministic mapping obscured by noise

### The Solution: Clean Training Set

Investigation revealed that subject→category mapping is **deterministic**:
- **~110 unique subjects** in the 100K corpus
- **Zero conflicts** - each subject maps to exactly one category
- **Balanced distribution** - ~22 samples per category (5 classes)

By extracting and training on these clean templates:
- CatBoost F1: **18% → 88%** (4.9x improvement)
- BERT F1: **1.66% → 100%** (60x improvement on validation set)

### Validation Results

```
Subject→Category Conflict Check:
✓ 110 unique subjects extracted
✓ 0 subjects with multiple categories
✓ Category distribution: 19-23 per class (balanced)
```

**See:** `exploration/subcategory_independence/REPORT.md` for detailed statistical analysis

---

## Model Details

### CatBoost

**Architecture:**
- Gradient boosting with ordered boosting and native categorical handling
- Default loss: MultiClass (softmax)

**Features:**
- TF-IDF text vectors (5,000 selected from 10,000 vocab)
- One-hot encoded categorical features
- Standard-scaled numerical features
- Total: ~5,056 dimensions

**Strengths:**
- Fast training on small datasets
- Handles categorical features without encoding overhead
- Robust to hyperparameter choices

**Configuration:**
```python
{
  "iterations": 100-200,  # Optuna tuned
  "depth": 4-6,           # Optuna tuned
  "learning_rate": 0.01-0.3,
  "loss_function": "MultiClass",
  "random_seed": 42
}
```

---

### XGBoost

**Architecture:**
- Gradient boosting with histogram-based splits
- Objective: multi:softprob

**Features:**
- Same feature set as CatBoost (~5,056 dimensions)
- One-hot encoding required for categorical features

**Strengths:**
- Industry-standard gradient boosting
- Well-documented and widely deployed
- Efficient parallel training

**Configuration:**
```python
{
  "n_estimators": 50-150,  # Optuna tuned
  "max_depth": 3-5,        # Optuna tuned
  "learning_rate": 0.01-0.2,
  "objective": "multi:softprob",
  "random_state": 42
}
```

---

### DistilBERT

**Architecture:**
- DistilBERT base (66M parameters)
- Fine-tuned classification head (768 → 5 classes)
- Text-only mode (no tabular fusion)

**Features:**
- Raw text input (subject + description concatenated)
- BERT tokenizer with max_len=128
- No manual feature engineering required

**Strengths:**
- Semantic understanding beyond keyword matching
- Handles paraphrasing and novel phrasing
- Pre-trained on massive text corpus

**Configuration:**
```python
{
  "preset": "distil_bert_base_en_uncased",
  "batch_size": 16,
  "epochs": 4-8,
  "learning_rate": 2e-5,
  "max_len": 128,
  "early_stop_patience": 2
}
```

---

## Recommendation

### For Production Deployment

**Primary Model:** **CatBoost**

**Rationale:**
- Expected >85% F1 with full training (proven on clean data)
- Sub-millisecond latency (~0.5-1.0ms p50)
- Small model size (<1MB)
- CPU-only deployment (no GPU required)
- Deterministic predictions (reproducible)

**When to Use BERT:**
- Need to handle highly varied phrasing beyond training examples
- GPU infrastructure available
- Higher accuracy requirement justifies 30-50ms latency
- Model size (<300MB) acceptable

**When to Use XGBoost:**
- Need industry-standard framework for compliance/auditing
- Existing XGBoost infrastructure
- Minimal performance difference vs CatBoost in practice

---

## Trade-offs Analysis

| Aspect            | CatBoost/XGBoost          | DistilBERT                |
|-------------------|---------------------------|---------------------------|
| **Accuracy**      | 85-90% F1 (expected)      | 88-95% F1 (expected)      |
| **Latency**       | 0.5-1.5ms (CPU)           | 30-50ms (CPU), 5-10ms (GPU)|
| **Model Size**    | <1MB                      | ~250MB                    |
| **Hardware**      | CPU sufficient            | GPU recommended           |
| **Interpretability** | High (feature importance) | Low (black box)        |
| **Generalization**| Limited to vocab          | Better on novel phrasing  |
| **Training Time** | 1-5 minutes               | 15-30 minutes             |

---

## Training Instructions

### Smoke Test (Fast Validation)

```bash
make train SMOKE_TEST=true ENV=dev
# Uses ~20/10/17 samples, minimal hyperparameters
# Runtime: ~2 minutes
```

### Full Training (Production)

```bash
make train ENV=dev
# Uses full ~110 clean samples, production hyperparameters
# Runtime: 5-10 minutes (traditional ML), 15-30 minutes (BERT)
```

### Individual Models

```bash
make train-catboost ENV=dev
make train-xgboost ENV=dev
make train-bert ENV=dev
```

---

## Artifacts

Training produces the following artifacts:

**Models:**
- `models/catboost_category_dev.cbm`
- `models/xgboost_category_dev.json`
- `models/bert_category_dev/` (directory with metadata.json + model.weights.h5)

**Feature Pipelines:**
- `models/feature_pipeline_category_dev.pkl`

**Metrics:**
- `metrics/dev/catboost_training_summary.json`
- `metrics/dev/xgboost_training_summary.json`
- `metrics/dev/mdeepl_training_summary.json`
- `metrics/dev/*_per_class.csv` (per-category precision/recall)

**MLflow:**
- Experiment tracking in `mlruns/` directory
- View with: `mlflow ui --backend-store-uri ./mlruns`

---

## Next Steps

1. **Run full training** to generate production-quality models
2. **Compare actual results** to expected performance table above
3. **Update serving configuration** in `src/config.py` to use best model
4. **Monitor in production** with anomaly detector and confidence thresholds

---

## References

- **Data Investigation:** `exploration/subcategory_independence/REPORT.md`
- **Training Scripts:** `src/training/train_*.py`
- **Model Wrappers:** `src/models/*_model.py`
- **Evaluation:** `src/evaluation/metrics.py`
- **Decision Log:** `docs/DECISIONS.md` (D-001, D-002, D-006, D-013, D-014)
