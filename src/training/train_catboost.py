"""Train CatBoost category classifier on clean split data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import config, mlflow_utils
from src.evaluation.latency import benchmark_model
from src.evaluation.metrics import compute_all_metrics
from src.features.pipeline import FeaturePipeline
from src.models.catboost_model import CatBoostTicketClassifier

try:  # pragma: no cover - runtime dependency
    import optuna
except Exception:  # pragma: no cover - runtime dependency
    optuna = None


def _summary_path() -> Path:
    return config.METRICS_DIR / "catboost_training_summary.json"


def _per_class_path() -> Path:
    return config.METRICS_DIR / "catboost_per_class.csv"


def _load_split_data(parquet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Clean training data not found: {parquet_path}\n\n"
            f"Please run the data pipeline first:\n"
            f"  ENV={config.ENV} make data-pipeline\n"
        )

    df = pd.read_parquet(parquet_path)
    if "split" not in df.columns:
        raise ValueError("Expected 'split' column in clean training parquet")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/val/test splits must all be non-empty")

    return train_df, val_df, test_df


def _sample_if_needed(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=config.SPLIT_SEED)


def _optuna_trials_count() -> int:
    if config.IS_INFRA_DEV:
        return 2
    if config.SMOKE_TEST:
        return 10
    return 20


def _run_optuna(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
) -> dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is not installed")

    search_space = config.CATBOOST_SEARCH_SPACE

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "iterations": trial.suggest_int("iterations", *search_space["iterations"]),
            "depth": trial.suggest_int("depth", *search_space["depth"]),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                *search_space["learning_rate"],
                log=True,
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *search_space["l2_leaf_reg"]),
            "random_strength": trial.suggest_float(
                "random_strength",
                *search_space["random_strength"],
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature",
                *search_space["bagging_temperature"],
            ),
            "border_count": trial.suggest_int("border_count", *search_space["border_count"]),
        }

        model = CatBoostTicketClassifier(params=trial_params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        y_pred = model.predict_label_ids(X_val)
        return float(f1_score(y_val, y_pred, average="weighted"))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.SPLIT_SEED),
    )
    study.optimize(
        objective,
        n_trials=_optuna_trials_count(),
        timeout=config.OPTUNA_TIMEOUT_S,
        show_progress_bar=False,
    )

    summary = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "timeout_seconds": config.OPTUNA_TIMEOUT_S,
    }

    config.CATBOOST_OPTUNA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CATBOOST_OPTUNA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_catboost_model(*, enable_optuna: bool | None = None, classifier: str = "category") -> dict[str, Any]:
    """Train CatBoost model and return training summary payload."""
    config.ensure_directories()

    use_optuna = config.TRADML_ENABLE_OPTUNA if enable_optuna is None else enable_optuna

    train_df, val_df, test_df = _load_split_data(config.CLEAN_TRAINING_PARQUET_PATH)

    if config.SMOKE_TEST:
        train_df = _sample_if_needed(train_df, config.TRAIN_SAMPLES)
        val_df = _sample_if_needed(val_df, config.VAL_SAMPLES)

    # Load or create pipeline for the specified classifier
    pipeline_path = config.FEATURE_PIPELINE_PATHS[classifier]
    if pipeline_path.exists():
        pipeline = FeaturePipeline.load(pipeline_path)
    else:
        pipeline = FeaturePipeline.create_for_classifier(classifier)
        pipeline.fit(train_df)

    X_train, y_train = pipeline.get_catboost_features(train_df)
    X_val, y_val = pipeline.get_catboost_features(val_df)
    X_test, y_test = pipeline.get_catboost_features(test_df)
    class_names = pipeline.label_encoder.classes_.tolist()

    params: dict[str, Any] = {}
    optuna_summary: dict[str, Any] | None = None
    if use_optuna:
        optuna_summary = _run_optuna(X_train, y_train, X_val, y_val)
        params.update(optuna_summary["best_params"])

    model = CatBoostTicketClassifier(params=params, label_classes=class_names)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_pred = model.predict_label_ids(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_all_metrics(y_test, y_pred, y_proba, class_names)

    latency = benchmark_model(
        lambda: model.predict_proba(X_test[:1]),
        n_warmup=3 if config.SMOKE_TEST else 10,
        n_iter=20 if config.SMOKE_TEST else 200,
    )

    pipeline.save(config.FEATURE_PIPELINE_PATHS[classifier])
    if classifier == "category":
        pipeline.tfidf.save(config.TFIDF_VECTORIZER_PATH)
    model.save(config.CATBOOST_MODEL_PATHS[classifier])

    per_class_df = metrics["per_class_metrics"]
    per_class_path = _per_class_path()
    per_class_df.to_csv(per_class_path, index=False)

    summary = {
        "model": "catboost",
        "data": {
            "path": str(config.CLEAN_TRAINING_PARQUET_PATH),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "smoke_test": config.SMOKE_TEST,
        },
        "params": model.params,
        "optuna": optuna_summary,
        "metrics": {
            "f1_weighted": float(metrics["f1_weighted"]),
            "f1_macro": float(metrics["f1_macro"]),
            "accuracy": float(metrics["accuracy"]),
            "recall_macro": float(metrics["recall_macro"]),
            "precision_macro": float(metrics["precision_macro"]),
            "ece": float(metrics["ece"]),
        },
        "latency": latency,
        "artifacts": {
            "model": str(config.CATBOOST_MODEL_PATHS[classifier]),
            "feature_pipeline": str(config.FEATURE_PIPELINE_PATHS[classifier]),
            "per_class_metrics": str(per_class_path),
        },
    }

    summary_path = _summary_path()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with mlflow_utils.start_run("train-catboost-category"):
        mlflow_utils.log_tags(
            {
                "model": "catboost",
                "target": "category",
                "data_version": config.CLEAN_TRAINING_PARQUET_PATH.name,
                "smoke_test": config.SMOKE_TEST,
            }
        )
        mlflow_utils.log_params(model.params)
        mlflow_utils.log_params(
            {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "optuna_enabled": use_optuna,
            }
        )
        mlflow_utils.log_metrics(
            {
                "test_f1_weighted": summary["metrics"]["f1_weighted"],
                "test_f1_macro": summary["metrics"]["f1_macro"],
                "test_accuracy": summary["metrics"]["accuracy"],
                "test_recall_macro": summary["metrics"]["recall_macro"],
                "test_precision_macro": summary["metrics"]["precision_macro"],
                "test_ece": summary["metrics"]["ece"],
                "latency_p50_ms": float(latency["single_sample_p50_ms"]),
                "latency_p95_ms": float(latency["single_sample_p95_ms"]),
            }
        )
        mlflow_utils.log_artifact(str(summary_path))
        mlflow_utils.log_artifact(str(per_class_path))
        if use_optuna and config.CATBOOST_OPTUNA_SUMMARY_PATH.exists():
            mlflow_utils.log_artifact(str(config.CATBOOST_OPTUNA_SUMMARY_PATH))

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CatBoost classifier")
    parser.add_argument("--classifier", choices=["category", "priority", "sentiment"], default="category")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna tuning")
    parser.add_argument("--no-optuna", action="store_true", help="Disable Optuna tuning")
    args = parser.parse_args(argv)

    use_optuna = config.TRADML_ENABLE_OPTUNA
    if args.optuna:
        use_optuna = True
    if args.no_optuna:
        use_optuna = False

    summary = train_catboost_model(enable_optuna=use_optuna, classifier=args.classifier)
    print(
        "CatBoost training complete: "
        f"f1_weighted={summary['metrics']['f1_weighted']:.4f}, "
        f"accuracy={summary['metrics']['accuracy']:.4f}"
    )
    print(f"Summary: {_summary_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
