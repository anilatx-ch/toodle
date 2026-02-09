"""Tests for XGBoost model wrapper."""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("xgboost")

from src.models.xgboost_model import XGBoostTicketClassifier


@pytest.fixture
def tiny_multiclass_data():
    X = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.1, 0.0],
                [0.0, 1.0, 0.1],
                [0.1, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    return X, y


def test_xgboost_fit_predict_shapes(tiny_multiclass_data):
    X, y = tiny_multiclass_data
    model = XGBoostTicketClassifier(
        params={
            "n_estimators": 12,
            "max_depth": 2,
            "learning_rate": 0.2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "verbosity": 0,
        },
        label_classes=["a", "b", "c"],
    )

    model.fit(X, y)
    proba = model.predict_proba(X)
    pred_ids = model.predict_label_ids(X)

    assert proba.shape == (len(y), 3)
    assert pred_ids.shape == (len(y),)
    assert set(pred_ids.tolist()) <= {0, 1, 2}


def test_xgboost_save_and_load(tmp_path: Path, tiny_multiclass_data):
    X, y = tiny_multiclass_data
    model = XGBoostTicketClassifier(
        params={
            "n_estimators": 12,
            "max_depth": 2,
            "learning_rate": 0.2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "verbosity": 0,
        },
        label_classes=["a", "b", "c"],
    )
    model.fit(X, y)

    model_path = tmp_path / "xgboost_category.json"
    model.save(model_path)

    loaded = XGBoostTicketClassifier.load(model_path)
    loaded_proba = loaded.predict_proba(X)

    assert loaded_proba.shape == (len(y), 3)
    assert (tmp_path / "xgboost_category.json.meta.json").exists()


def test_xgboost_feature_importance_dataframe(tiny_multiclass_data):
    X, y = tiny_multiclass_data
    model = XGBoostTicketClassifier(
        params={
            "n_estimators": 12,
            "max_depth": 2,
            "learning_rate": 0.2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "verbosity": 0,
        }
    )
    model.fit(X, y)

    importance = model.get_feature_importance()
    assert list(importance.columns) == ["feature", "importance"]
