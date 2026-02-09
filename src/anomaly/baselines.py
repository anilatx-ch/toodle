"""Baseline builders for anomaly detection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src import config
from src.anomaly.detector import AnomalyBaseline, CategoryBaseline


def compute_baseline_from_predictions(
    predictions_df: pd.DataFrame,
    category_col: str = "predicted_category",
    confidence_col: str = "category_confidence",
) -> AnomalyBaseline:
    """Build baseline from historical prediction logs."""
    if predictions_df.empty:
        return AnomalyBaseline(created_at=datetime.now(UTC))

    stats_df = (
        predictions_df.groupby(category_col)
        .agg(
            volume_count=(category_col, "count"),
            confidence_mean=(confidence_col, "mean"),
            confidence_std=(confidence_col, "std"),
        )
        .reset_index()
    )
    stats_df["confidence_std"] = stats_df["confidence_std"].fillna(0.0)

    baselines: dict[str, CategoryBaseline] = {}
    for row in stats_df.itertuples(index=False):
        category = str(getattr(row, category_col))
        baselines[category] = CategoryBaseline(
            category=category,
            confidence_mean=float(row.confidence_mean),
            confidence_std=float(row.confidence_std),
            volume_count=int(row.volume_count),
            sample_count=int(row.volume_count),
            last_updated=datetime.now(UTC),
        )

    total = len(predictions_df)
    distribution = {
        str(category): float(count / total)
        for category, count in predictions_df[category_col].value_counts().items()
    }
    return AnomalyBaseline(
        category_baselines=baselines,
        overall_category_distribution=distribution,
        total_predictions=total,
        created_at=datetime.now(UTC),
    )


def compute_baseline_from_training_data(
    parquet_path: Path = config.CLEAN_TRAINING_PARQUET_PATH,
    category_col: str = "category",
) -> AnomalyBaseline:
    """Bootstrap baseline from clean training data."""
    if not parquet_path.exists():
        return AnomalyBaseline(created_at=datetime.now(UTC))

    df = pd.read_parquet(parquet_path)
    if category_col not in df.columns or df.empty:
        return AnomalyBaseline(created_at=datetime.now(UTC))

    category_counts = df[category_col].value_counts()
    total = int(category_counts.sum())
    baselines: dict[str, CategoryBaseline] = {}
    for category, count in category_counts.items():
        baselines[str(category)] = CategoryBaseline(
            category=str(category),
            confidence_mean=0.75,
            confidence_std=0.15,
            volume_count=int(count),
            sample_count=int(count),
            last_updated=datetime.now(UTC),
        )

    distribution = {
        str(category): float(count / total)
        for category, count in category_counts.items()
    }
    return AnomalyBaseline(
        category_baselines=baselines,
        overall_category_distribution=distribution,
        total_predictions=total,
        created_at=datetime.now(UTC),
    )


def build_and_save_baseline(
    source: str = "training",
    output_path: Path | None = None,
) -> AnomalyBaseline:
    """Build and persist anomaly baseline."""
    path = output_path or config.ANOMALY_BASELINE_PATH
    if source == "training":
        baseline = compute_baseline_from_training_data()
    else:
        raise ValueError(f"Unsupported baseline source: {source!r}")
    baseline.save(path)
    return baseline
