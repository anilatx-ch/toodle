"""Fit and persist feature pipeline artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.features.pipeline import FeaturePipeline


def main() -> int:
    config.ensure_directories()
    df = pd.read_parquet(config.CLEAN_TRAINING_PARQUET_PATH)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if config.SMOKE_TEST:
        train_df = train_df.head(config.TRAIN_SAMPLES)
        val_df = val_df.head(config.VAL_SAMPLES)

    pipeline = FeaturePipeline.create_for_classifier("category")
    pipeline.fit(train_df)

    X_train, y_train = pipeline.get_catboost_features(train_df)
    X_val, y_val = pipeline.get_catboost_features(val_df)

    pipeline.save(config.FEATURE_PIPELINE_PATHS["category"])
    pipeline.tfidf.save(config.TFIDF_VECTORIZER_PATH)

    print(f"train shape={X_train.shape}, val shape={X_val.shape}")
    print(f"label classes={len(pipeline.label_encoder.classes_)}")
    print(f"train labels shape={y_train.shape}, val labels shape={y_val.shape}")
    print("Saved feature artifacts in models/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
