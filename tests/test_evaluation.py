"""Tests for evaluation metrics and analysis."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from src.evaluation.analysis import generate_error_analysis
from src.evaluation.metrics import compute_all_metrics, expected_calibration_error


CLASS_NAMES = [
    "Technical Issue",
    "Billing",
    "Account Management",
    "Feature Request",
    "General Inquiry",
]


def test_f1_perfect_predictions():
    """Test F1 score with perfect predictions."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = y_true.copy()
    y_proba = np.eye(5)
    metrics = compute_all_metrics(y_true, y_pred, y_proba, CLASS_NAMES)
    assert metrics["f1_weighted"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_ece_perfect_calibration():
    """Test ECE with perfectly calibrated predictions."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_proba = np.eye(5)
    assert expected_calibration_error(y_true, y_proba) < 1e-6


def test_compute_all_metrics_keys():
    """Test that all expected metrics are computed."""
    y_true = np.array([0, 1, 2, 1, 3])
    y_pred = np.array([0, 2, 2, 1, 3])
    y_proba = np.array(
        [
            [0.8, 0.1, 0.05, 0.03, 0.02],
            [0.1, 0.3, 0.5, 0.05, 0.05],
            [0.2, 0.2, 0.5, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.05, 0.05],
            [0.1, 0.1, 0.1, 0.6, 0.1],
        ]
    )
    metrics = compute_all_metrics(y_true, y_pred, y_proba, CLASS_NAMES)
    expected_keys = {
        "f1_weighted",
        "f1_macro",
        "accuracy",
        "recall_macro",
        "precision_macro",
        "ece",
        "per_class_metrics",
        "confusion_matrix",
    }
    assert expected_keys <= set(metrics.keys())
    assert isinstance(metrics["per_class_metrics"], pd.DataFrame)
    assert len(metrics["per_class_metrics"]) == 5


def test_error_analysis_sample_count():
    """Test that error analysis returns correct number of samples."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([1, 1, 2, 0, 2, 0])
    y_proba = np.array(
        [
            [0.2, 0.6, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.05, 0.05],
            [0.1, 0.2, 0.6, 0.05, 0.05],
            [0.6, 0.2, 0.1, 0.05, 0.05],
            [0.1, 0.2, 0.6, 0.05, 0.05],
            [0.6, 0.2, 0.1, 0.05, 0.05],
        ]
    )
    df = pd.DataFrame(
        {
            "ticket_id": [f"T{i}" for i in range(len(y_true))],
            "subject": [f"subject {i}" for i in range(len(y_true))],
            "description": [f"description {i}" for i in range(len(y_true))],
        }
    )
    samples = generate_error_analysis(df, y_true, y_pred, y_proba, CLASS_NAMES, n_samples=2)
    assert len(samples) == 2
    assert "true_category" in samples[0]
    assert "predicted_category" in samples[0]
    assert "confidence" in samples[0]
    assert "top_3_predictions" in samples[0]


def test_confusion_matrix_shape():
    """Test confusion matrix shape matches number of classes."""
    y_true = np.array([0, 1, 2, 3, 4, 0, 1])
    y_pred = np.array([0, 1, 2, 3, 4, 1, 1])
    y_proba = np.eye(5)[y_pred]
    metrics = compute_all_metrics(y_true, y_pred, y_proba, CLASS_NAMES)
    cm = metrics["confusion_matrix"]
    assert cm.shape == (5, 5)


def test_per_class_metrics_format():
    """Test per-class metrics DataFrame format."""
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 2, 2])
    y_proba = np.eye(5)[y_pred]
    metrics = compute_all_metrics(y_true, y_pred, y_proba, CLASS_NAMES)
    per_class = metrics["per_class_metrics"]

    assert len(per_class) == 5
    assert list(per_class.columns) == ["class", "precision", "recall", "f1", "support"]
    assert list(per_class["class"]) == CLASS_NAMES
