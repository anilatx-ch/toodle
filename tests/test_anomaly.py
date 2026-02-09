"""Unit tests for anomaly and volume analysis modules."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.anomaly.baselines import (
    build_and_save_baseline,
    compute_baseline_from_predictions,
    compute_baseline_from_training_data,
)
from src.anomaly.detector import AnomalyBaseline, AnomalyDetector, CategoryBaseline
from src.anomaly.volume_analyzer import analyze_volume_patterns, js_divergence


def test_category_baseline_zscore() -> None:
    baseline = CategoryBaseline(
        category="Technical Issue",
        confidence_mean=0.80,
        confidence_std=0.10,
        volume_count=100,
        sample_count=100,
        last_updated=datetime.now(UTC),
    )
    assert baseline.confidence_zscore(0.80) == pytest.approx(0.0)
    assert baseline.confidence_zscore(0.60) == pytest.approx(-2.0)


def test_anomaly_detector_flags_low_confidence() -> None:
    baseline = AnomalyBaseline(
        category_baselines={
            "Technical Issue": CategoryBaseline(
                category="Technical Issue",
                confidence_mean=0.85,
                confidence_std=0.10,
                volume_count=100,
                sample_count=100,
                last_updated=datetime.now(UTC),
            )
        },
        overall_category_distribution={},
        total_predictions=100,
        created_at=datetime.now(UTC),
    )
    detector = AnomalyDetector(baseline=baseline)
    normal = detector.analyze_prediction("Technical Issue", 0.80)
    assert not normal.is_confidence_anomaly
    low = detector.analyze_prediction("Technical Issue", 0.50)
    assert low.is_confidence_anomaly
    assert low.confidence_zscore < -2.0


def test_anomaly_baseline_roundtrip(tmp_path) -> None:
    original = AnomalyBaseline(
        category_baselines={
            "Technical Issue": CategoryBaseline(
                category="Technical Issue",
                confidence_mean=0.80,
                confidence_std=0.10,
                volume_count=20,
                sample_count=20,
                last_updated=datetime.now(UTC),
            )
        },
        overall_category_distribution={"Technical Issue": 1.0},
        total_predictions=20,
        created_at=datetime.now(UTC),
    )
    path = tmp_path / "baseline.json"
    original.save(path)
    loaded = AnomalyBaseline.load(path)
    assert loaded is not None
    assert loaded.total_predictions == 20
    assert loaded.category_baselines["Technical Issue"].confidence_mean == pytest.approx(0.80)


def test_compute_baseline_from_predictions() -> None:
    df = pd.DataFrame(
        {
            "predicted_category": ["A", "A", "B"],
            "category_confidence": [0.9, 0.8, 0.7],
        }
    )
    baseline = compute_baseline_from_predictions(df)
    assert baseline.total_predictions == 3
    assert set(baseline.category_baselines) == {"A", "B"}
    assert baseline.overall_category_distribution["A"] == pytest.approx(2 / 3)


def test_compute_baseline_from_training_data(tmp_path) -> None:
    path = tmp_path / "clean.parquet"
    pd.DataFrame({"category": ["A", "A", "B", "B", "B"]}).to_parquet(path, index=False)
    baseline = compute_baseline_from_training_data(path)
    assert baseline.total_predictions == 5
    assert baseline.category_baselines["B"].sample_count == 3


def test_build_and_save_baseline(tmp_path) -> None:
    output = tmp_path / "baseline.json"
    baseline = build_and_save_baseline(output_path=output)
    assert output.exists()
    assert baseline.total_predictions >= 0


def test_js_divergence_properties() -> None:
    p = np.asarray([0.5, 0.5])
    q = np.asarray([0.5, 0.5])
    r = np.asarray([0.9, 0.1])
    assert js_divergence(p, q) < 1e-6
    assert js_divergence(p, r) > 0.0


def test_analyze_volume_patterns() -> None:
    baseline_df = pd.DataFrame({"predicted_category": ["A"] * 90 + ["B"] * 10})
    current_df = pd.DataFrame({"predicted_category": ["A"] * 20 + ["B"] * 80})
    report = analyze_volume_patterns(current_df, baseline_df)
    assert report.is_distribution_anomaly
    assert len(report.anomaly_summary) > 0


def test_analyze_volume_patterns_requires_non_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        analyze_volume_patterns(pd.DataFrame(), pd.DataFrame())
