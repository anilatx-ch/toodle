"""Tests for report generation script."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def mock_training_summaries(tmp_path):
    """Create mock training summary files."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    catboost_summary = {
        "metadata": {"model": "catboost", "timestamp": "2024-01-01T00:00:00"},
        "params": {"iterations": 500, "depth": 6},
        "metrics": {
            "f1_weighted": 0.88,
            "f1_macro": 0.87,
            "accuracy": 0.89,
            "recall_macro": 0.86,
            "precision_macro": 0.88,
            "ece": 0.12,
        },
        "latency": {
            "single_sample_p50_ms": 1.2,
            "single_sample_p95_ms": 2.5,
            "throughput_samples_per_sec": 800,
        },
    }

    xgboost_summary = {
        "metadata": {"model": "xgboost", "timestamp": "2024-01-01T00:00:00"},
        "params": {"n_estimators": 500, "max_depth": 6},
        "metrics": {
            "f1_weighted": 0.86,
            "f1_macro": 0.85,
            "accuracy": 0.87,
            "recall_macro": 0.84,
            "precision_macro": 0.86,
            "ece": 0.14,
        },
        "latency": {
            "single_sample_p50_ms": 1.5,
            "single_sample_p95_ms": 3.0,
            "throughput_samples_per_sec": 650,
        },
    }

    bert_summary = {
        "metadata": {"model": "bert", "timestamp": "2024-01-01T00:00:00"},
        "params": {"epochs": 4, "batch_size": 16, "learning_rate": 2e-5},
        "metrics": {
            "f1_weighted": 0.92,
            "f1_macro": 0.91,
            "accuracy": 0.93,
            "ece": 0.08,
        },
        "latency": {
            "single_sample_p50_ms": 45.0,
            "single_sample_p95_ms": 65.0,
            "throughput_samples_per_sec": 22,
        },
    }

    (metrics_dir / "catboost_training_summary.json").write_text(
        json.dumps(catboost_summary), encoding="utf-8"
    )
    (metrics_dir / "xgboost_training_summary.json").write_text(
        json.dumps(xgboost_summary), encoding="utf-8"
    )
    (metrics_dir / "mdeepl_training_summary.json").write_text(
        json.dumps(bert_summary), encoding="utf-8"
    )

    return tmp_path


def _import_generate_report():
    root = Path(__file__).parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts import generate_report

    return generate_report


def test_report_generation_with_mock_data(mock_training_summaries, tmp_path):
    """Generate report and verify docs/MODEL.md is not modified."""
    metrics_dir = mock_training_summaries / "metrics"
    generate_report = _import_generate_report()

    root = Path(__file__).parents[1]
    model_doc = root / "docs" / "MODEL.md"
    model_doc_before = model_doc.read_text(encoding="utf-8")
    model_doc_mtime_before = model_doc.stat().st_mtime_ns

    with mock.patch("src.config.METRICS_DIR", metrics_dir):
        with mock.patch("src.config.MDEEPL_TRAINING_SUMMARY_PATH", metrics_dir / "mdeepl_training_summary.json"):
            with mock.patch("src.config.MODEL_COMPARISON_PATH", tmp_path / "report.md"):
                with mock.patch("src.config.CATBOOST_MODEL_PATH", tmp_path / "catboost.cbm"):
                    with mock.patch("src.config.XGBOOST_MODEL_PATH", tmp_path / "xgboost.json"):
                        with mock.patch("src.config.BERT_MODEL_DIR", tmp_path / "bert"):
                            with mock.patch("src.config.ENV", "test"):
                                (tmp_path / "catboost.cbm").write_bytes(b"x" * 1024 * 100)
                                (tmp_path / "xgboost.json").write_bytes(b"x" * 1024 * 50)
                                bert_dir = tmp_path / "bert"
                                bert_dir.mkdir()
                                (bert_dir / "model.weights.h5").write_bytes(b"x" * 1024 * 1024 * 250)

                                result = generate_report.main()

    assert result == 0

    report_path = tmp_path / "report.md"
    assert report_path.exists()
    report_content = report_path.read_text(encoding="utf-8")

    assert "Model Comparison Report" in report_content
    assert "CatBoost" in report_content
    assert "XGBoost" in report_content
    assert "DistilBERT" in report_content
    assert "0.8800" in report_content
    assert "0.8600" in report_content
    assert "0.9200" in report_content
    assert "1.20" in report_content
    assert "45.00" in report_content
    assert "0.10" in report_content
    assert "250.00" in report_content

    # Regression guard: report generation must not mutate MODEL.md.
    assert model_doc.read_text(encoding="utf-8") == model_doc_before
    assert model_doc.stat().st_mtime_ns == model_doc_mtime_before


def test_report_with_missing_models(tmp_path):
    """Report generation should still succeed when model artifacts are missing."""
    generate_report = _import_generate_report()
    metrics_dir = tmp_path / "missing_metrics"
    report_path = tmp_path / "report_missing.md"

    with mock.patch("src.config.METRICS_DIR", metrics_dir):
        with mock.patch("src.config.MDEEPL_TRAINING_SUMMARY_PATH", metrics_dir / "mdeepl_training_summary.json"):
            with mock.patch("src.config.MODEL_COMPARISON_PATH", report_path):
                with mock.patch("src.config.CATBOOST_MODEL_PATH", tmp_path / "missing_catboost.cbm"):
                    with mock.patch("src.config.XGBOOST_MODEL_PATH", tmp_path / "missing_xgboost.json"):
                        with mock.patch("src.config.BERT_MODEL_DIR", tmp_path / "missing_bert_dir"):
                            with mock.patch("src.config.ENV", "test"):
                                result = generate_report.main()

    assert result == 0
    assert report_path.exists()
    report_content = report_path.read_text(encoding="utf-8")
    assert "No models have been trained yet" in report_content
    assert "make train-tradml" in report_content
