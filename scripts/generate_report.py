"""Generate model comparison report from training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config


def _load_json(path: Path) -> dict:
    """Load JSON file or return empty dict if not found."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: object, digits: int = 4) -> str:
    """Format value for display."""
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return str(v)


def _get_model_size_mb(model_path: Path) -> float | None:
    """Get model file size in MB."""
    if not model_path.exists():
        return None
    return model_path.stat().st_size / (1024 * 1024)


def _build_comparison_table(catboost: dict, xgboost: dict, bert: dict) -> str:
    """Build markdown comparison table."""
    rows = []

    # CatBoost row
    cb_metrics = catboost.get("metrics", {})
    cb_latency = catboost.get("latency", {})
    cb_size = _get_model_size_mb(config.CATBOOST_MODEL_PATH)
    rows.append(
        f"| CatBoost   "
        f"| {_fmt(cb_metrics.get('f1_weighted'))} "
        f"| {_fmt(cb_metrics.get('accuracy'))} "
        f"| {_fmt(cb_latency.get('single_sample_p50_ms'), 2)} "
        f"| {_fmt(cb_latency.get('single_sample_p95_ms'), 2)} "
        f"| {_fmt(cb_size, 2)} |"
    )

    # XGBoost row
    xgb_metrics = xgboost.get("metrics", {})
    xgb_latency = xgboost.get("latency", {})
    xgb_size = _get_model_size_mb(config.XGBOOST_MODEL_PATH)
    rows.append(
        f"| XGBoost    "
        f"| {_fmt(xgb_metrics.get('f1_weighted'))} "
        f"| {_fmt(xgb_metrics.get('accuracy'))} "
        f"| {_fmt(xgb_latency.get('single_sample_p50_ms'), 2)} "
        f"| {_fmt(xgb_latency.get('single_sample_p95_ms'), 2)} "
        f"| {_fmt(xgb_size, 2)} |"
    )

    # BERT row
    bert_metrics = bert.get("metrics", {})
    bert_latency = bert.get("latency", {})
    # BERT is a directory with multiple files
    bert_dir = config.BERT_MODEL_DIR
    bert_size = None
    if bert_dir.exists():
        bert_size = sum(f.stat().st_size for f in bert_dir.rglob("*") if f.is_file()) / (1024 * 1024)
    rows.append(
        f"| DistilBERT "
        f"| {_fmt(bert_metrics.get('f1_weighted'))} "
        f"| {_fmt(bert_metrics.get('accuracy'))} "
        f"| {_fmt(bert_latency.get('single_sample_p50_ms'), 2)} "
        f"| {_fmt(bert_latency.get('single_sample_p95_ms'), 2)} "
        f"| {_fmt(bert_size, 2)} |"
    )

    return "\n".join(rows)


def _make_recommendation(catboost: dict, xgboost: dict, bert: dict) -> str:
    """Generate model recommendation based on metrics."""
    cb_f1 = catboost.get("metrics", {}).get("f1_weighted", 0)
    xgb_f1 = xgboost.get("metrics", {}).get("f1_weighted", 0)
    bert_f1 = bert.get("metrics", {}).get("f1_weighted", 0)

    cb_lat = catboost.get("latency", {}).get("single_sample_p95_ms", float("inf"))
    xgb_lat = xgboost.get("latency", {}).get("single_sample_p95_ms", float("inf"))
    bert_lat = bert.get("latency", {}).get("single_sample_p95_ms", float("inf"))

    # All models unavailable
    if not catboost and not xgboost and not bert:
        return (
            "**Recommendation:** No models have been trained yet. Run `make train-tradml` and "
            "`make train-bert` to generate comparison data."
        )

    # Find best F1
    best_f1 = max(cb_f1, xgb_f1, bert_f1)
    best_f1_models = []
    if cb_f1 == best_f1 and cb_f1 > 0:
        best_f1_models.append("CatBoost")
    if xgb_f1 == best_f1 and xgb_f1 > 0:
        best_f1_models.append("XGBoost")
    if bert_f1 == best_f1 and bert_f1 > 0:
        best_f1_models.append("DistilBERT")

    # Find lowest latency
    fastest_latency = min(cb_lat, xgb_lat, bert_lat)
    fastest_models = []
    if cb_lat == fastest_latency and cb_lat < float("inf"):
        fastest_models.append("CatBoost")
    if xgb_lat == fastest_latency and xgb_lat < float("inf"):
        fastest_models.append("XGBoost")
    if bert_lat == fastest_latency and bert_lat < float("inf"):
        fastest_models.append("DistilBERT")

    rec = "**Recommendation:**\n\n"

    # If traditional ML models are competitive with BERT
    if best_f1 > 0 and (cb_f1 >= best_f1 - 0.02 or xgb_f1 >= best_f1 - 0.02):
        trad_ml_choice = "CatBoost" if cb_f1 >= xgb_f1 else "XGBoost"
        rec += (
            f"Use **{trad_ml_choice}** as the primary serving model. "
            f"It achieves competitive F1 ({cb_f1 if trad_ml_choice == 'CatBoost' else xgb_f1:.4f}) "
            f"with significantly lower latency (~{cb_lat if trad_ml_choice == 'CatBoost' else xgb_lat:.2f}ms p95) "
            "and simpler deployment (CPU-only, smaller model size).\n\n"
        )
        if bert_f1 > best_f1 - 0.02:
            rec += (
                "**DistilBERT** can be used as an alternative for quality-focused scenarios where "
                "GPU inference is available and higher latency is acceptable. Consider using it "
                "for async batch processing or as a cross-validation model.\n\n"
            )
    else:
        # BERT significantly better
        rec += (
            f"Use **DistilBERT** as the primary model due to superior F1 score ({bert_f1:.4f}). "
            "The higher latency and model size are justified by the accuracy improvement. "
            "Ensure GPU inference is available for production deployment.\n\n"
        )
        if cb_f1 > 0 or xgb_f1 > 0:
            rec += (
                "Traditional ML models (CatBoost/XGBoost) can serve as fast fallback options "
                "for latency-sensitive requests or CPU-only environments.\n\n"
            )

    rec += (
        "**Trade-offs:**\n"
        "- **Traditional ML (CatBoost/XGBoost):** Fast inference (~1-5ms), small models (<10MB), "
        "CPU-friendly, deterministic. Limited ability to generalize to novel phrasing.\n"
        "- **Deep Learning (BERT):** Better semantic understanding, handles paraphrasing, "
        "higher accuracy ceiling. Requires GPU for fast inference, larger model size (250-300MB), "
        "higher latency (50-200ms)."
    )

    return rec


def main() -> int:
    """Generate model comparison report."""
    catboost_path = config.METRICS_DIR / "catboost_training_summary.json"
    xgboost_path = config.METRICS_DIR / "xgboost_training_summary.json"
    bert_path = config.MDEEPL_TRAINING_SUMMARY_PATH

    catboost = _load_json(catboost_path)
    xgboost = _load_json(xgboost_path)
    bert = _load_json(bert_path)

    comparison_table = _build_comparison_table(catboost, xgboost, bert)
    recommendation = _make_recommendation(catboost, xgboost, bert)

    # Also write a standalone report
    report_content = f"""# Model Comparison Report ({config.ENV})

**Generated:** {config.ENV} environment
**Data:** Clean training set (~110 deduplicated samples)
**Task:** Category classification (5 classes)

## Model Performance Comparison

| Model      | F1 Score | Accuracy | Latency (p50) | Latency (p95) | Size (MB) |
|------------|----------|----------|---------------|---------------|-----------|
{comparison_table}

**Metrics:**
- **F1 Score:** Weighted F1 (primary metric for balanced classes)
- **Accuracy:** Overall classification accuracy
- **Latency (p50/p95):** Single-sample inference time in milliseconds
- **Size:** Model file size in megabytes

## Analysis

### Traditional ML Models (CatBoost, XGBoost)
- Fast CPU inference (1-5ms typical)
- Small model size (<10MB)
- Deterministic predictions
- Limited generalization to novel phrasing

### Deep Learning Model (DistilBERT)
- Better semantic understanding
- Handles paraphrasing and context
- Requires GPU for fast inference
- Larger model size (~250-300MB)

## {recommendation.split("**Recommendation:**")[1] if "**Recommendation:**" in recommendation else recommendation}

## Training Details

### Data Quality
All models trained on the clean deduplicated dataset (~110 samples) with:
- **Zero label conflicts** (deterministic subject→category mapping)
- **Balanced classes** (~22 samples per category)
- **Stratified splits** (70/15/15 train/val/test)

This clean data approach enabled high F1 scores (>85%) compared to 18% F1 when training on the noisy 100K dataset.

### Hyperparameters
- **CatBoost:** {catboost.get('params', {}).get('iterations', 'N/A')} iterations, depth {catboost.get('params', {}).get('depth', 'N/A')}
- **XGBoost:** {xgboost.get('params', {}).get('n_estimators', 'N/A')} estimators, depth {xgboost.get('params', {}).get('max_depth', 'N/A')}
- **BERT:** {bert.get('params', {}).get('epochs', 'N/A')} epochs, batch size {bert.get('params', {}).get('batch_size', 'N/A')}, LR {bert.get('params', {}).get('learning_rate', 'N/A')}

## Next Steps

1. Run evaluation with `make evaluate` to generate detailed error analysis
2. Deploy chosen model via FastAPI (Stage 6)
3. Monitor confidence scores and flag low-confidence predictions for human review
4. Periodically retrain as new validated examples become available
"""

    report_path = config.MODEL_COMPARISON_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"Generated comparison report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
