"""Train DistilBERT category classifier on clean subject/category splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from src import config, mlflow_utils
from src.evaluation.latency import benchmark_model
from src.evaluation.metrics import compute_all_metrics
from src.models.bert_model import BertClassifier

try:  # pragma: no cover - runtime dependency
    import keras_nlp
    import tensorflow as tf
except Exception:  # pragma: no cover - runtime dependency
    keras_nlp = None
    tf = None

_TEXT_SEPARATOR = " [SEP] "


def _require_runtime() -> None:
    if tf is None or keras_nlp is None:
        raise RuntimeError("TensorFlow and keras-nlp are required for BERT training")


def _compose_text(df: pd.DataFrame) -> list[str]:
    subject = df["subject"].fillna("").astype(str)
    description = df["description"].fillna("").astype(str)
    return subject.str.cat(description, sep=_TEXT_SEPARATOR).str.strip().tolist()


def _load_splits(parquet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing clean training data at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if config.SMOKE_TEST:
        train_df = train_df.head(config.TRAIN_SAMPLES).copy()
        val_df = val_df.head(config.VAL_SAMPLES).copy()
        test_df = test_df.head(config.VAL_SAMPLES).copy()

    return train_df, val_df, test_df


def _make_dataset(
    texts: list[str],
    tabular: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    training: bool,
):
    inputs = {"text_input": np.asarray(texts, dtype=object)}
    if tabular.shape[1] > 0:
        inputs["tabular_input"] = tabular.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((inputs, labels.astype(np.int32)))
    if training:
        ds = ds.shuffle(min(len(texts), 2048), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if tf is not None:  # pragma: no cover - runtime dependency
    class ValF1Callback(tf.keras.callbacks.Callback):
        """Compute validation weighted F1 each epoch."""

        def __init__(self, val_texts: list[str], val_tabular: np.ndarray, val_labels: np.ndarray):
            super().__init__()
            self.val_texts = val_texts
            self.val_tabular = val_tabular
            self.val_labels = val_labels
            self.history: list[float] = []

        def on_epoch_end(self, epoch, logs=None):
            predictions = self.model.predict(  # noqa: WPS437 - Keras callback API
                {
                    "text_input": np.asarray(self.val_texts, dtype=object),
                    **(
                        {"tabular_input": self.val_tabular}
                        if self.val_tabular.shape[1] > 0
                        else {}
                    ),
                },
                verbose=0,
            )
            pred_ids = np.asarray(predictions).argmax(axis=1)
            val_f1 = float(f1_score(self.val_labels, pred_ids, average="weighted"))
            self.history.append(val_f1)
            logs = logs or {}
            logs["val_f1_weighted"] = val_f1
            mlflow_utils.log_metric("val_f1_weighted", val_f1, step=epoch)


else:
    class ValF1Callback:  # pragma: no cover - runtime dependency
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("TensorFlow is required for ValF1Callback")


def _download_preset() -> None:
    _require_runtime()
    keras_nlp.models.DistilBertPreprocessor.from_preset(
        config.BERT_PRESET, sequence_length=config.BERT_MAX_LEN
    )
    keras_nlp.models.DistilBertBackbone.from_preset(config.BERT_PRESET)


def train(
    parquet_path: Path,
    model_dir: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Train BERT category classifier and return summary payload."""
    _require_runtime()
    config.ensure_directories()

    train_df, val_df, test_df = _load_splits(parquet_path)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train/val/test splits must all be non-empty")

    train_texts = _compose_text(train_df)
    val_texts = _compose_text(val_df)
    test_texts = _compose_text(test_df)

    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_df["category"].astype(str))
    val_y = label_encoder.transform(val_df["category"].astype(str))
    test_y = label_encoder.transform(test_df["category"].astype(str))
    label_names = label_encoder.classes_.tolist()

    train_tabular = np.zeros((len(train_texts), 0), dtype=np.float32)
    val_tabular = np.zeros((len(val_texts), 0), dtype=np.float32)
    test_tabular = np.zeros((len(test_texts), 0), dtype=np.float32)

    classifier = BertClassifier(classifier="category", label_classes=label_names)
    model = classifier.build_model(n_tabular_features=0, n_classes=len(label_names))

    train_ds = _make_dataset(
        train_texts,
        train_tabular,
        train_y,
        batch_size=config.BERT_BATCH_SIZE,
        training=True,
    )
    val_ds = _make_dataset(
        val_texts,
        val_tabular,
        val_y,
        batch_size=config.BERT_BATCH_SIZE,
        training=False,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.BERT_EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        ),
        ValF1Callback(val_texts, val_tabular, val_y),
    ]

    start = time.perf_counter()
    with mlflow_utils.start_run("train-bert-category"):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.BERT_EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )
        train_seconds = float(time.perf_counter() - start)

        classifier.model = model
        classifier.save(model_dir)

        test_proba = classifier.predict_proba(test_texts, test_tabular)
        test_pred = test_proba.argmax(axis=1)
        metrics = compute_all_metrics(test_y, test_pred, test_proba, label_names)

        per_class = metrics["per_class_metrics"]
        per_class.to_csv(config.PER_CLASS_MDEEPL_PATH, index=False)

        latency = benchmark_model(
            lambda: classifier.predict_proba(test_texts[:1], test_tabular[:1]),
            n_warmup=3 if config.SMOKE_TEST else 10,
            n_iter=10 if config.SMOKE_TEST else 100,
        )

        summary = {
            "classifier": "category",
            "model_dir": str(model_dir),
            "dataset_path": str(parquet_path),
            "epochs_ran": len(history.history.get("loss", [])),
            "training_time_seconds": train_seconds,
            "n_train_samples": len(train_df),
            "n_val_samples": len(val_df),
            "n_test_samples": len(test_df),
            "f1_weighted": float(metrics["f1_weighted"]),
            "f1_macro": float(metrics["f1_macro"]),
            "accuracy": float(metrics["accuracy"]),
            "recall_macro": float(metrics["recall_macro"]),
            "precision_macro": float(metrics["precision_macro"]),
            "ece": float(metrics["ece"]),
            "latency": latency,
        }

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        mlflow_utils.log_params(
            {
                "model_type": "distilbert",
                "classifier": "category",
                "preset": config.BERT_PRESET,
                "batch_size": config.BERT_BATCH_SIZE,
                "epochs": config.BERT_EPOCHS,
                "early_stop_patience": config.BERT_EARLY_STOP_PATIENCE,
                "learning_rate": config.BERT_LR,
                "weight_decay": config.BERT_WEIGHT_DECAY,
                "max_len": config.BERT_MAX_LEN,
            }
        )
        mlflow_utils.log_metrics(
            {
                "test_f1_weighted": summary["f1_weighted"],
                "test_f1_macro": summary["f1_macro"],
                "test_accuracy": summary["accuracy"],
                "test_ece": summary["ece"],
                "training_time_seconds": summary["training_time_seconds"],
                "latency_p50_ms": latency["single_sample_p50_ms"],
                "latency_p95_ms": latency["single_sample_p95_ms"],
            }
        )
        mlflow_utils.log_artifact(str(summary_path))
        mlflow_utils.log_artifact(str(config.PER_CLASS_MDEEPL_PATH))

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DistilBERT category classifier")
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=config.CLEAN_TRAINING_PARQUET_PATH,
        help="Path to clean training parquet",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=config.BERT_MODEL_DIR,
        help="Output directory for BERT model",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=config.MDEEPL_TRAINING_SUMMARY_PATH,
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download preset assets and exit",
    )
    args = parser.parse_args()

    if args.download_only:
        _download_preset()
        print(f"Downloaded preset: {config.BERT_PRESET}")
        return 0

    summary = train(args.parquet_path, args.model_dir, args.summary_path)
    print(
        "BERT training complete: "
        f"f1_weighted={summary['f1_weighted']:.4f}, accuracy={summary['accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

