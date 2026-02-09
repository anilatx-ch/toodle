"""Tests for feature pipeline."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from src.features.pipeline import FeaturePipeline


def test_pipeline_catboost_output_shapes(feature_df):
    train_df = feature_df[feature_df.split == "train"]
    pipeline = FeaturePipeline.create_for_classifier("category")
    pipeline.fit(train_df)

    X, y = pipeline.get_catboost_features(train_df)
    transformed = pipeline.transform(train_df)
    expected_feature_count = transformed["tfidf"].shape[1] + transformed["tabular"].shape[1]
    assert X.shape[0] == len(train_df)
    assert X.shape[1] == expected_feature_count
    assert transformed["tfidf"].shape[1] > 0
    assert y.shape[0] == len(train_df)


def test_pipeline_bert_output_types(feature_df):
    train_df = feature_df[feature_df.split == "train"]
    pipeline = FeaturePipeline.create_for_classifier("category")
    pipeline.fit(train_df)

    texts, tabular, y = pipeline.get_bert_features(train_df)
    assert isinstance(texts, list)
    assert isinstance(tabular, np.ndarray)
    assert tabular.shape[0] == len(train_df)
    assert y.shape[0] == len(train_df)


def test_label_encoding_roundtrip(feature_df):
    train_df = feature_df[feature_df.split == "train"]
    pipeline = FeaturePipeline.create_for_classifier("category")
    pipeline.fit(train_df)

    y = pipeline.label_encoder.transform(train_df["category"])
    decoded = pipeline.decode_labels(y)
    assert decoded.tolist() == train_df["category"].tolist()


def test_priority_pipeline_fit(feature_df):
    train_df = feature_df[feature_df.split == "train"].copy()
    train_df["priority"] = np.where(train_df.index % 2 == 0, "high", "low")
    train_df["severity"] = np.where(train_df.index % 3 == 0, "P1", "P3")

    priority_pipeline = FeaturePipeline.create_for_classifier("priority")
    priority_pipeline.fit(train_df)
    _, _, priority_y = priority_pipeline.get_bert_features(train_df)
    assert priority_y is not None
    assert priority_y.shape[0] == len(train_df)


def test_category_pipeline_does_not_require_priority_column():
    train_rows = np.array(
        [
            ["subject a", "description a", "Technical Issue"],
            ["subject b", "description b", "Feature Request"],
            ["subject c", "description c", "Security"],
        ],
        dtype=object,
    )
    df = pd.DataFrame(train_rows, columns=["subject", "description", "category"])

    category_pipeline = FeaturePipeline.create_for_classifier("category")
    category_pipeline.fit(df)
    _, y = category_pipeline.get_catboost_features(df)
    assert y is not None
    assert y.shape[0] == len(df)
