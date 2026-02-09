"""DistilBERT classifier wrapper with optional tabular fusion branch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config

try:  # pragma: no cover - runtime dependency
    import keras_nlp
    import tensorflow as tf
except Exception:  # pragma: no cover - runtime dependency
    keras_nlp = None
    tf = None


def _default_labels(classifier: str) -> list[str]:
    labels = config.CLASSIFIER_LABELS.get(classifier)
    if labels is None:
        raise ValueError(f"Unsupported classifier={classifier!r}")
    return list(labels)


@dataclass
class BertClassifier:
    """DistilBERT classifier for text-only or text+tabular inference."""

    classifier: str = "category"
    preset: str = config.BERT_PRESET
    label_classes: list[str] | None = None
    model: object | None = None
    n_tabular_features: int = 0

    def __post_init__(self) -> None:
        if self.label_classes is None:
            self.label_classes = _default_labels(self.classifier)

    def _ensure_runtime(self) -> None:
        if tf is None or keras_nlp is None:
            raise RuntimeError("TensorFlow and keras-nlp are required for BertClassifier")

    def _compile(self, model):
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.BERT_LR,
            weight_decay=config.BERT_WEIGHT_DECAY,
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            jit_compile=False,
        )
        return model

    def build_model(
        self,
        n_tabular_features: int = 0,
        n_classes: int | None = None,
        compile_model: bool = True,
    ):
        """Build DistilBERT model graph."""
        self._ensure_runtime()

        if n_classes is None:
            n_classes = len(self.label_classes)

        preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
            self.preset,
            sequence_length=config.BERT_MAX_LEN,
        )
        backbone = keras_nlp.models.DistilBertBackbone.from_preset(self.preset)

        text_input = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        encoded = backbone(preprocessor(text_input))
        sequence_output = encoded["sequence_output"] if isinstance(encoded, dict) else encoded
        pooled_text = tf.keras.layers.GlobalAveragePooling1D(name="pooled_text")(sequence_output)

        merged = pooled_text
        inputs = {"text_input": text_input}
        if n_tabular_features > 0:
            tabular_input = tf.keras.Input(
                shape=(n_tabular_features,),
                dtype=tf.float32,
                name="tabular_input",
            )
            tabular_branch = tf.keras.layers.Dense(config.BERT_TABULAR_DENSE, activation="relu")(
                tabular_input
            )
            tabular_branch = tf.keras.layers.Dropout(config.BERT_DROPOUT)(tabular_branch)
            merged = tf.keras.layers.Concatenate()([merged, tabular_branch])
            inputs["tabular_input"] = tabular_input

        merged = tf.keras.layers.Dense(config.BERT_HIDDEN_1, activation="relu")(merged)
        merged = tf.keras.layers.Dropout(config.BERT_DROPOUT)(merged)
        merged = tf.keras.layers.Dense(config.BERT_HIDDEN_2, activation="relu")(merged)
        merged = tf.keras.layers.Dropout(config.BERT_DROPOUT)(merged)

        outputs = tf.keras.layers.Dense(
            int(n_classes),
            activation="softmax",
            dtype="float32",
            name="probs",
        )(merged)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.n_tabular_features = int(n_tabular_features)
        self.model = self._compile(model) if compile_model else model
        return self.model

    def _build_inputs(self, texts: list[str], tabular: np.ndarray | None) -> dict[str, np.ndarray]:
        text_array = np.asarray(texts, dtype=object)
        inputs: dict[str, np.ndarray] = {"text_input": text_array}
        if self.n_tabular_features > 0:
            if tabular is None:
                raise ValueError("tabular features are required for this model")
            tabular_array = np.asarray(tabular, dtype=np.float32)
            inputs["tabular_input"] = tabular_array
        return inputs

    def predict_proba(self, texts: list[str], tabular: np.ndarray | None = None) -> np.ndarray:
        """Predict class probabilities for each sample."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        outputs = self.model(self._build_inputs(texts, tabular), training=False)
        if hasattr(outputs, "numpy"):
            outputs = outputs.numpy()
        return np.asarray(outputs, dtype=np.float32)

    def predict_label_ids(self, texts: list[str], tabular: np.ndarray | None = None) -> np.ndarray:
        """Predict integer class labels."""
        return self.predict_proba(texts, tabular).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        """Save model weights and metadata."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        weights_path = model_dir / "model.weights.h5"
        self.model.save_weights(str(weights_path))

        output_shape = self.model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        metadata = {
            "classifier": self.classifier,
            "preset": self.preset,
            "label_classes": self.label_classes,
            "n_tabular_features": int(self.n_tabular_features),
            "n_classes": int(output_shape[-1]),
            "weights_ref": weights_path.name,
        }
        with open(model_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @staticmethod
    def resolve_model_artifact(path: str | Path) -> Path:
        """Resolve weights file path from model directory."""
        model_dir = Path(path)
        if model_dir.is_file():
            return model_dir

        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            weights_ref = metadata.get("weights_ref", "")
            if isinstance(weights_ref, str):
                weights_path = model_dir / weights_ref
                if weights_path.exists():
                    return weights_path

        fallback = model_dir / "model.weights.h5"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"No Bert model artifact found under: {model_dir}")

    @classmethod
    def load(cls, path: str | Path) -> "BertClassifier":
        """Load classifier from metadata and weights."""
        model_dir = Path(path)
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata at {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        classifier_name = str(metadata.get("classifier", "category"))
        raw_labels = metadata.get("label_classes")
        label_classes = (
            list(raw_labels)
            if isinstance(raw_labels, list) and raw_labels
            else _default_labels(classifier_name)
        )

        classifier = cls(
            classifier=classifier_name,
            preset=str(metadata.get("preset", config.BERT_PRESET)),
            label_classes=label_classes,
        )

        n_tabular_features = int(metadata.get("n_tabular_features", 0))
        n_classes = int(metadata.get("n_classes", len(classifier.label_classes)))
        weights_path = cls.resolve_model_artifact(model_dir)

        classifier.build_model(
            n_tabular_features=n_tabular_features,
            n_classes=n_classes,
            compile_model=False,
        )
        classifier.model.load_weights(str(weights_path))
        classifier.model = classifier._compile(classifier.model)
        return classifier
