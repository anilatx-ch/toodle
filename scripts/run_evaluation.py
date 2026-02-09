"""Run full model evaluation, explainability, and latency benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, mlflow_utils
from src.evaluation.analysis import (
    confidence_analysis,
    generate_confusion_clusters,
    generate_error_analysis,
)
from src.evaluation.latency import benchmark_model
from src.evaluation.metrics import compute_all_metrics
from src.features.pipeline import FeaturePipeline


def _plot_cm(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels, fmt="d")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_calibration(
    y_true: np.ndarray, y_proba: np.ndarray, out_path: Path, title: str
) -> None:
    """Plot calibration curve."""
    conf = np.max(y_proba, axis=1)
    preds = np.argmax(y_proba, axis=1)
    correct = (preds == y_true).astype(int)
    prob_true, prob_pred = calibration_curve(correct, conf, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    ax.set_title(title)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _save_latency_plot(latency_dict: dict[str, dict], out_path: Path) -> None:
    """Plot latency comparison across models."""
    names = list(latency_dict.keys())
    p50 = [latency_dict[k]["single_sample_p50_ms"] for k in names]
    p95 = [latency_dict[k]["single_sample_p95_ms"] for k in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, p50, width, label="p50")
    ax.bar(x + width / 2, p95, width, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("ms")
    ax.set_title("Latency comparison")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_feature_importance(model, out_path: Path, model_name: str, top_k: int = 30) -> None:
    """Plot feature importance for traditional ML models."""
    importance = model.get_feature_importance().head(top_k).copy()
    if importance.empty:
        print(f"[WARNING] No feature importance available for {model_name}")
        return

    importance = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance["feature"], importance["importance"], color="tab:blue")
    ax.set_title(f"{model_name} feature importance (top {top_k})")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    """Run evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--model", choices=["catboost", "xgboost", "bert", "all"], default="all")
    parser.add_argument("--classifier", choices=["category", "sentiment"], default="category")
    parser.add_argument("--latency-only", action="store_true")
    args = parser.parse_args()

    config.ensure_directories()
    pipeline = FeaturePipeline.load(config.FEATURE_PIPELINE_PATHS[args.classifier])
    df = pd.read_parquet(config.SPLIT_PARQUET_PATH)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    if config.SMOKE_TEST:
        test_df = test_df.head(200).copy()

    class_names = pipeline.label_encoder.classes_.tolist()
    evaluation_payload = {}
    evaluation_artifacts = []

    # Determine output path based on classifier
    if args.classifier == "sentiment":
        output_path = config.METRICS_DIR / "sentiment_evaluation_summary.json"
    else:
        output_path = config.EVALUATION_SUMMARY_PATH

    # CatBoost evaluation
    if args.model in ["catboost", "all"] and not args.latency_only:
        try:
            from src.models.catboost_model import CatBoostTicketClassifier

            catboost_model = CatBoostTicketClassifier.load(config.CATBOOST_MODEL_PATHS[args.classifier])
            catboost_model.label_classes = class_names
            X_test, y_test = pipeline.get_catboost_features(test_df)
            y_pred_cat = catboost_model.predict_label_ids(X_test)
            y_proba_cat = catboost_model.predict_proba(X_test)

            metrics_cat = compute_all_metrics(y_test, y_pred_cat, y_proba_cat, class_names)
            evaluation_payload["catboost"] = {
                k: v
                for k, v in metrics_cat.items()
                if k not in ["confusion_matrix", "per_class_metrics"]
            }

            per_class_path = config.METRICS_DIR / "catboost_per_class.csv"
            metrics_cat["per_class_metrics"].to_csv(per_class_path, index=False)
            evaluation_artifacts.append(per_class_path)

            cm_path = config.FIGURES_DIR / "confusion_matrix_catboost.png"
            _plot_cm(metrics_cat["confusion_matrix"], class_names, "CatBoost Confusion Matrix", cm_path)
            evaluation_artifacts.append(cm_path)

            cal_path = config.FIGURES_DIR / "calibration_catboost.png"
            _plot_calibration(y_test, y_proba_cat, cal_path, "CatBoost Calibration")
            evaluation_artifacts.append(cal_path)

            fi_path = config.FIGURES_DIR / "feature_importance_catboost.png"
            _plot_feature_importance(catboost_model, fi_path, "CatBoost")
            evaluation_artifacts.append(fi_path)

            errors = generate_error_analysis(
                test_df, y_test, y_pred_cat, y_proba_cat, class_names, n_samples=50
            )
            clusters = generate_confusion_clusters(metrics_cat["confusion_matrix"], class_names)
            conf_stats = confidence_analysis(y_test, y_pred_cat, y_proba_cat)

            error_path = config.DIAGNOSTICS_DIR / "catboost_error_analysis.json"
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump({"errors": errors, "clusters": clusters, "confidence": conf_stats}, f, indent=2)
            evaluation_artifacts.append(error_path)

            print(f"CatBoost F1: {metrics_cat['f1_weighted']:.4f}")
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[WARNING] CatBoost evaluation skipped: {e}")

    # XGBoost evaluation
    if args.model in ["xgboost", "all"] and not args.latency_only:
        try:
            from src.models.xgboost_model import XGBoostTicketClassifier

            xgboost_model = XGBoostTicketClassifier.load(config.XGBOOST_MODEL_PATHS[args.classifier])
            xgboost_model.label_classes = class_names
            X_test, y_test = pipeline.get_catboost_features(test_df)
            y_pred_xgb = xgboost_model.predict_label_ids(X_test)
            y_proba_xgb = xgboost_model.predict_proba(X_test)

            metrics_xgb = compute_all_metrics(y_test, y_pred_xgb, y_proba_xgb, class_names)
            evaluation_payload["xgboost"] = {
                k: v
                for k, v in metrics_xgb.items()
                if k not in ["confusion_matrix", "per_class_metrics"]
            }

            per_class_path = config.METRICS_DIR / "xgboost_per_class.csv"
            metrics_xgb["per_class_metrics"].to_csv(per_class_path, index=False)
            evaluation_artifacts.append(per_class_path)

            cm_path = config.FIGURES_DIR / "confusion_matrix_xgboost.png"
            _plot_cm(metrics_xgb["confusion_matrix"], class_names, "XGBoost Confusion Matrix", cm_path)
            evaluation_artifacts.append(cm_path)

            cal_path = config.FIGURES_DIR / "calibration_xgboost.png"
            _plot_calibration(y_test, y_proba_xgb, cal_path, "XGBoost Calibration")
            evaluation_artifacts.append(cal_path)

            fi_path = config.FIGURES_DIR / "feature_importance_xgboost.png"
            _plot_feature_importance(xgboost_model, fi_path, "XGBoost")
            evaluation_artifacts.append(fi_path)

            print(f"XGBoost F1: {metrics_xgb['f1_weighted']:.4f}")
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[WARNING] XGBoost evaluation skipped: {e}")

    # BERT evaluation
    if args.model in ["bert", "all"] and not args.latency_only:
        try:
            from src.models.bert_model import BertClassifier

            bert_model = BertClassifier.load(config.BERT_MODEL_DIRS[args.classifier])
            bert_model.label_classes = class_names
            test_texts, test_tabular, y_test = pipeline.get_bert_features(test_df)
            y_proba_bert = bert_model.predict_proba(test_texts, test_tabular)
            y_pred_bert = y_proba_bert.argmax(axis=1)

            metrics_bert = compute_all_metrics(y_test, y_pred_bert, y_proba_bert, class_names)
            evaluation_payload["bert"] = {
                k: v
                for k, v in metrics_bert.items()
                if k not in ["confusion_matrix", "per_class_metrics"]
            }

            per_class_path = config.METRICS_DIR / "bert_per_class.csv"
            metrics_bert["per_class_metrics"].to_csv(per_class_path, index=False)
            evaluation_artifacts.append(per_class_path)

            cm_path = config.FIGURES_DIR / "confusion_matrix_bert.png"
            _plot_cm(metrics_bert["confusion_matrix"], class_names, "BERT Confusion Matrix", cm_path)
            evaluation_artifacts.append(cm_path)

            cal_path = config.FIGURES_DIR / "calibration_bert.png"
            _plot_calibration(y_test, y_proba_bert, cal_path, "BERT Calibration")
            evaluation_artifacts.append(cal_path)

            print(f"BERT F1: {metrics_bert['f1_weighted']:.4f}")
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[WARNING] BERT evaluation skipped: {e}")

    # Save evaluation summary
    if evaluation_payload:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_payload, f, indent=2)

        with mlflow_utils.start_run("evaluation-test-metrics"):
            mlflow_utils.log_params({"env": config.ENV, "smoke_test": config.SMOKE_TEST})
            for model_name, vals in evaluation_payload.items():
                for metric_name in [
                    "f1_weighted",
                    "f1_macro",
                    "accuracy",
                    "recall_macro",
                    "precision_macro",
                    "ece",
                ]:
                    if metric_name in vals:
                        mlflow_utils.log_metric(f"test_{model_name}_{metric_name}", float(vals[metric_name]))
            mlflow_utils.log_artifact(str(output_path))
            for artifact in evaluation_artifacts:
                if artifact.exists():
                    mlflow_utils.log_artifact(str(artifact))

    print("Evaluation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
