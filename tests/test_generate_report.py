"""Tests for report generation script."""

import json
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


def test_report_generation_with_mock_data(mock_training_summaries, tmp_path):
    """Test that report generation works with mock data."""
    import sys
    from pathlib import Path

    # Add project root to path
    root = Path(__file__).parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Mock config paths
    metrics_dir = mock_training_summaries / "metrics"
    with mock.patch("src.config.METRICS_DIR", metrics_dir):
        with mock.patch("src.config.MDEEPL_TRAINING_SUMMARY_PATH", metrics_dir / "mdeepl_training_summary.json"):
            with mock.patch("src.config.MODEL_COMPARISON_PATH", tmp_path / "report.md"):
                with mock.patch("src.config.CATBOOST_MODEL_PATH", tmp_path / "catboost.cbm"):
                    with mock.patch("src.config.XGBOOST_MODEL_PATH", tmp_path / "xgboost.json"):
                        with mock.patch("src.config.BERT_MODEL_DIR", tmp_path / "bert"):
                            with mock.patch("src.config.ENV", "test"):
                                # Import after mocking
                                from scripts import generate_report

                            # Create mock model files
                            (tmp_path / "catboost.cbm").write_bytes(b"x" * 1024 * 100)  # 100KB
                            (tmp_path / "xgboost.json").write_bytes(b"x" * 1024 * 50)  # 50KB
                            bert_dir = tmp_path / "bert"
                            bert_dir.mkdir()
                            (bert_dir / "model.weights.h5").write_bytes(b"x" * 1024 * 1024 * 250)  # 250MB

                            # Create mock MODEL.md
                            model_doc = root / "docs" / "MODEL.md"
                            original_content = ""
                            if model_doc.exists():
                                original_content = model_doc.read_text(encoding="utf-8")

                            with mock.patch.object(Path, "exists", return_value=True):
                                with mock.patch.object(
                                    Path,
                                    "read_text",
                                    return_value="# Existing content\n\n"
                                    "| Model      | F1 Score | Accuracy | Latency (p50) | Latency (p95) | Size (MB) |\n"
                                    "|------------|----------|----------|---------------|---------------|-----------|",
                                ):
                                    result = generate_report.main()

                            # Verify success
                            assert result == 0

                            # Verify report was created
                            report_path = tmp_path / "report.md"
                            assert report_path.exists()

                            report_content = report_path.read_text(encoding="utf-8")

                            # Verify report contains expected sections
                            assert "Model Comparison Report" in report_content
                            assert "CatBoost" in report_content
                            assert "XGBoost" in report_content
                            assert "DistilBERT" in report_content

                            # Verify metrics are present
                            assert "0.8800" in report_content  # CatBoost F1
                            assert "0.8600" in report_content  # XGBoost F1
                            assert "0.9200" in report_content  # BERT F1

                            # Verify latencies are present
                            assert "1.20" in report_content  # CatBoost p50
                            assert "45.00" in report_content  # BERT p50

                            # Verify sizes are present
                            assert "0.10" in report_content  # CatBoost size
                            assert "250.00" in report_content  # BERT size


def test_report_with_missing_models():
    """Test report generation when no models are trained."""
    import sys
    from pathlib import Path

    root = Path(__file__).parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    metrics_dir = Path("/nonexistent/metrics")
    with mock.patch("src.config.METRICS_DIR", metrics_dir):
        with mock.patch("src.config.MDEEPL_TRAINING_SUMMARY_PATH", metrics_dir / "mdeepl_training_summary.json"):
            with mock.patch("src.config.MODEL_COMPARISON_PATH", Path("/tmp/report.md")):
                with mock.patch("src.config.CATBOOST_MODEL_PATH", Path("/nonexistent/model.cbm")):
                    with mock.patch("src.config.XGBOOST_MODEL_PATH", Path("/nonexistent/model.json")):
                        with mock.patch("src.config.BERT_MODEL_DIR", Path("/nonexistent/bert")):
                            with mock.patch("src.config.ENV", "test"):
                                from scripts import generate_report

                            # Mock Path operations
                            with mock.patch.object(Path, "exists", return_value=False):
                                with mock.patch.object(Path, "mkdir"):
                                    with mock.patch.object(Path, "write_text"):
                                        result = generate_report.main()

                            # Should succeed even with no models
                            assert result == 0
