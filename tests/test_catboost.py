"""Tests for CatBoost model wrapper."""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("catboost")

from src.models.catboost_model import CatBoostTicketClassifier


@pytest.fixture
def tiny_multiclass_data():
    X = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.2, 0.0],
                [0.0, 1.0, 0.2],
                [0.1, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    return X, y


def test_catboost_fit_predict_shapes(tiny_multiclass_data):
    X, y = tiny_multiclass_data
    model = CatBoostTicketClassifier(
        params={
            "iterations": 8,
            "depth": 2,
            "learning_rate": 0.2,
            "verbose": False,
            "allow_writing_files": False,
        },
        label_classes=["a", "b", "c"],
    )

    model.fit(X, y)
    proba = model.predict_proba(X)
    pred_ids = model.predict_label_ids(X)

    assert proba.shape == (len(y), 3)
    assert pred_ids.shape == (len(y),)
    assert set(pred_ids.tolist()) <= {0, 1, 2}


def test_catboost_save_and_load(tmp_path: Path, tiny_multiclass_data):
    X, y = tiny_multiclass_data
    model = CatBoostTicketClassifier(
        params={
            "iterations": 8,
            "depth": 2,
            "learning_rate": 0.2,
            "verbose": False,
            "allow_writing_files": False,
        },
        label_classes=["a", "b", "c"],
    )
    model.fit(X, y)

    model_path = tmp_path / "catboost_category.cbm"
    model.save(model_path)

    loaded = CatBoostTicketClassifier.load(model_path)
    loaded_proba = loaded.predict_proba(X)

    assert loaded_proba.shape == (len(y), 3)
    assert (tmp_path / "catboost_category.cbm.meta.json").exists()


def test_catboost_feature_importance_not_empty(tiny_multiclass_data):
    X, y = tiny_multiclass_data
    model = CatBoostTicketClassifier(
        params={
            "iterations": 8,
            "depth": 2,
            "learning_rate": 0.2,
            "verbose": False,
            "allow_writing_files": False,
        }
    )
    model.fit(X, y)

    importance = model.get_feature_importance()
    assert not importance.empty
    assert {"feature", "importance"} == set(importance.columns)
