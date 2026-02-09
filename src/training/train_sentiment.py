"""Train CatBoost sentiment classifier on feedback text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config, mlflow_utils
from src.evaluation.latency import benchmark_model
from src.evaluation.metrics import compute_all_metrics
from src.features.pipeline import FeaturePipeline
from src.models.catboost_model import CatBoostTicketClassifier


def _summary_path() -> Path:
    return config.SENTIMENT_TRAINING_SUMMARY_PATH


def _load_sentiment_rows(
    *,
    duckdb_path: Path,
    table: str = "mart_tickets_features",
    limit: int | None = None,
) -> pd.DataFrame:
    if not duckdb_path.exists():
        raise FileNotFoundError(
            f"Processed data DuckDB file not found: {duckdb_path}\n\n"
            f"Please run the data pipeline first:\n"
            f"  ENV={config.ENV} make data-pipeline\n"
        )

    query = f"""
        SELECT ticket_id, feedback_text, customer_sentiment
        FROM {table}
        WHERE NULLIF(TRIM(COALESCE(feedback_text, '')), '') IS NOT NULL
          AND NULLIF(TRIM(COALESCE(customer_sentiment, '')), '') IS NOT NULL
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    df["feedback_text"] = df["feedback_text"].astype(str)
    df["customer_sentiment"] = df["customer_sentiment"].astype(str)
    df = df[df["customer_sentiment"].isin(config.SENTIMENT_CLASSES)].copy()
    if df.empty:
        raise ValueError("No sentiment rows found with feedback_text and customer_sentiment")
    return df


def _split_data(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    random_state: int = config.SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    counts = df["customer_sentiment"].value_counts()
    keep_labels = sorted(counts[counts >= 2].index.tolist())
    filtered = df[df["customer_sentiment"].isin(keep_labels)].copy()
    if filtered.empty:
        raise ValueError("Sentiment dataset has no classes with at least 2 samples")

    train_df, val_df = train_test_split(
        filtered,
        test_size=val_fraction,
        random_state=random_state,
        stratify=filtered["customer_sentiment"],
    )
    return train_df, val_df, keep_labels


def _sample_if_needed(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=config.SPLIT_SEED).copy()


def train_sentiment_model(
    *,
    duckdb_path: Path = config.DUCKDB_PATH,
    table: str = "mart_tickets_features",
    limit: int | None = None,
) -> dict[str, Any]:
    """Train sentiment classifier and persist artifacts."""
    config.ensure_directories()

    df = _load_sentiment_rows(duckdb_path=duckdb_path, table=table, limit=limit)
    train_df, val_df, label_classes = _split_data(df)

    if config.SMOKE_TEST:
        train_df = _sample_if_needed(train_df, config.TRAIN_SAMPLES)
        val_df = _sample_if_needed(val_df, config.VAL_SAMPLES)

    pipeline = FeaturePipeline.create_for_classifier("sentiment")
    pipeline.label_encoder.fit(label_classes)
    pipeline.fit(train_df)

    X_train, y_train = pipeline.get_catboost_features(train_df)
    X_val, y_val = pipeline.get_catboost_features(val_df)
    if y_train is None or y_val is None:
        raise RuntimeError("Sentiment labels are missing")

    model = CatBoostTicketClassifier(label_classes=label_classes)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_pred = model.predict_label_ids(X_val)
    y_proba = model.predict_proba(X_val)
    metrics = compute_all_metrics(y_val, y_pred, y_proba, label_classes)
    latency = benchmark_model(
        lambda: model.predict_proba(X_val[:1]),
        n_warmup=3 if config.SMOKE_TEST else 10,
        n_iter=20 if config.SMOKE_TEST else 200,
    )

    model_path = config.CATBOOST_MODEL_PATHS["sentiment"]
    pipeline_path = config.FEATURE_PIPELINE_PATHS["sentiment"]
    model.save(model_path)
    pipeline.save(pipeline_path)

    summary = {
        "model": "catboost",
        "target": "sentiment",
        "data": {
            "duckdb_path": str(duckdb_path),
            "table": table,
            "rows_total": int(len(df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "smoke_test": config.SMOKE_TEST,
            "label_classes": label_classes,
        },
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
            "model": str(model_path),
            "feature_pipeline": str(pipeline_path),
        },
    }

    summary_path = _summary_path()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with mlflow_utils.start_run("train-catboost-sentiment"):
        mlflow_utils.log_tags(
            {
                "model": "catboost",
                "target": "sentiment",
                "data_version": f"{duckdb_path.name}:{table}",
                "smoke_test": config.SMOKE_TEST,
            }
        )
        mlflow_utils.log_params(
            {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "num_classes": len(label_classes),
            }
        )
        mlflow_utils.log_metrics(
            {
                "val_f1_weighted": summary["metrics"]["f1_weighted"],
                "val_f1_macro": summary["metrics"]["f1_macro"],
                "val_accuracy": summary["metrics"]["accuracy"],
                "val_recall_macro": summary["metrics"]["recall_macro"],
                "val_precision_macro": summary["metrics"]["precision_macro"],
                "val_ece": summary["metrics"]["ece"],
                "latency_p50_ms": float(latency["single_sample_p50_ms"]),
                "latency_p95_ms": float(latency["single_sample_p95_ms"]),
            }
        )
        mlflow_utils.log_artifact(str(summary_path))

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CatBoost sentiment classifier")
    parser.add_argument("--duckdb-path", type=str, default=str(config.DUCKDB_PATH))
    parser.add_argument("--table", type=str, default="mart_tickets_features")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    summary = train_sentiment_model(
        duckdb_path=Path(args.duckdb_path),
        table=args.table,
        limit=args.limit,
    )
    print(
        "Sentiment training complete: "
        f"f1_weighted={summary['metrics']['f1_weighted']:.4f}, "
        f"accuracy={summary['metrics']['accuracy']:.4f}"
    )
    print(f"Summary: {_summary_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

