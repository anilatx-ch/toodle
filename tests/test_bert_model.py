"""Tests for BERT classifier wrapper."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.models import bert_model
from src.models.bert_model import BertClassifier


class _FakeTensor:
    def __init__(self, name: str, shape):
        self.name = name
        self.shape = shape


class _FakeModel:
    def __init__(self, *, with_tabular: bool):
        self.inputs = [_FakeTensor("text_input:0", (None,))]
        if with_tabular:
            self.inputs.append(_FakeTensor("tabular_input:0", (None, 3)))
        self.output_shape = (None, 5)
        self.received_inputs = None

    def __call__(self, inputs, training=False):
        self.received_inputs = inputs
        return np.array([[0.1, 0.2, 0.3, 0.2, 0.2]], dtype=np.float32)

    def save_weights(self, path: str) -> None:
        with open(path, "wb") as handle:
            handle.write(b"weights")


def test_build_model_requires_runtime(monkeypatch):
    monkeypatch.setattr(bert_model, "tf", None)
    monkeypatch.setattr(bert_model, "keras_nlp", None)
    classifier = BertClassifier()
    with pytest.raises(RuntimeError, match="TensorFlow and keras-nlp are required"):
        classifier.build_model(n_tabular_features=0, n_classes=5)


def test_predict_proba_text_only_inputs():
    classifier = BertClassifier()
    classifier.n_tabular_features = 0
    classifier.model = _FakeModel(with_tabular=False)

    proba = classifier.predict_proba(["hello world"], None)
    assert proba.shape == (1, 5)
    assert set(classifier.model.received_inputs.keys()) == {"text_input"}


def test_predict_proba_requires_tabular_when_model_expects_it():
    classifier = BertClassifier()
    classifier.n_tabular_features = 3
    classifier.model = _FakeModel(with_tabular=True)

    with pytest.raises(ValueError, match="tabular features are required"):
        classifier.predict_proba(["hello world"], None)


def test_save_writes_metadata_and_weights(tmp_path):
    classifier = BertClassifier(classifier="category", label_classes=["A", "B", "C", "D", "E"])
    classifier.n_tabular_features = 0
    classifier.model = _FakeModel(with_tabular=False)

    model_dir = tmp_path / "bert_category"
    classifier.save(model_dir)

    metadata_path = model_dir / "metadata.json"
    weights_path = model_dir / "model.weights.h5"
    assert metadata_path.exists()
    assert weights_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["classifier"] == "category"
    assert metadata["n_tabular_features"] == 0
    assert metadata["n_classes"] == 5
    assert metadata["weights_ref"] == "model.weights.h5"


def test_resolve_model_artifact_prefers_metadata(tmp_path):
    model_dir = tmp_path / "bert_model"
    model_dir.mkdir(parents=True)
    target_weights = model_dir / "custom.weights.h5"
    target_weights.write_bytes(b"weights")
    metadata = {"weights_ref": "custom.weights.h5"}
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    resolved = BertClassifier.resolve_model_artifact(model_dir)
    assert resolved == target_weights

