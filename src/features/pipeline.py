"""Classifier-specific feature pipelines used by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from src import config
from src.features.categorical import CategoricalEncoder
from src.features.entities import EntityExtractor
from src.features.numerical import NumericalScaler
from src.features.preprocessing import augment_inference_features_df_vectorized
from src.features.text import TfidfFeaturizer, prepare_for_bert

_SEP_TOKEN = " [SEP] "


@dataclass(frozen=True)
class ClassifierFeatureSpec:
    classifier: str
    target_column: str
    fallback_target_columns: tuple[str, ...]
    label_classes: tuple[str, ...]
    text_tfidf_fields: tuple[str, ...]
    text_bert_fields: tuple[str, ...]
    categorical_fields: tuple[str, ...]
    ordinal_fields: tuple[str, ...]
    scaled_numeric_fields: tuple[str, ...]
    passthrough_numeric_fields: tuple[str, ...]
    include_entities: bool
    include_temporal: bool
    requires_category_hint: bool

    @staticmethod
    def for_classifier(classifier: str) -> "ClassifierFeatureSpec":
        if classifier not in config.CLASSIFIER_TARGET_COLUMN:
            raise ValueError(
                f"Unsupported classifier={classifier!r}. "
                f"Expected one of: {tuple(config.CLASSIFIER_TARGET_COLUMN.keys())}"
            )

        cfg = config.CLASSIFIER_FEATURE_CONFIGS[classifier]
        target_column = config.CLASSIFIER_TARGET_COLUMN[classifier]
        fallback_target_columns: tuple[str, ...] = ()
        if classifier == "sentiment":
            fallback_target_columns = ("sentiment",)

        return ClassifierFeatureSpec(
            classifier=classifier,
            target_column=target_column,
            fallback_target_columns=fallback_target_columns,
            label_classes=tuple(config.CLASSIFIER_LABELS[classifier]),
            text_tfidf_fields=tuple(cfg["text_tfidf_fields"]),
            text_bert_fields=tuple(cfg["text_bert_fields"]),
            categorical_fields=tuple(cfg["categorical_fields"]),
            ordinal_fields=tuple(cfg["ordinal_fields"]),
            scaled_numeric_fields=tuple(cfg["scaled_numeric_fields"]),
            passthrough_numeric_fields=tuple(cfg["passthrough_numeric_fields"]),
            include_entities=bool(cfg["include_entities"]),
            include_temporal=bool(cfg["include_temporal"]),
            requires_category_hint=bool(cfg["requires_category_hint"]),
        )


def _combine_text_fields(
    df: pd.DataFrame, fields: tuple[str, ...], *, separator: str, placeholder: str = ""
) -> pd.Series:
    if not fields:
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    series_parts: list[pd.Series] = []
    for field in fields:
        if field in df.columns:
            values = df[field]
        else:
            values = pd.Series([placeholder] * len(df), index=df.index)
        series_parts.append(values.fillna(placeholder).astype(str))

    combined = series_parts[0]
    for part in series_parts[1:]:
        combined = combined.str.cat(part, sep=separator)
    return combined.str.strip()


def _ensure_category_hint(df: pd.DataFrame) -> pd.DataFrame:
    if "category_hint" in df.columns:
        return df

    category_hint = pd.Series([""] * len(df), index=df.index)
    if "predicted_category" in df.columns:
        category_hint = df["predicted_category"].fillna("")
    elif "category" in df.columns:
        category_hint = df["category"].fillna("")

    out = df.copy()
    out["category_hint"] = category_hint.astype(str)
    return out


def _missing_columns(df: pd.DataFrame, required: set[str]) -> list[str]:
    return [column for column in sorted(required) if column not in df.columns]


@dataclass
class FeaturePipeline:
    tfidf: TfidfFeaturizer
    categorical: CategoricalEncoder
    numerical: NumericalScaler
    entities: EntityExtractor
    label_encoder: LabelEncoder
    spec: ClassifierFeatureSpec

    @staticmethod
    def create_for_classifier(classifier: str) -> "FeaturePipeline":
        spec = ClassifierFeatureSpec.for_classifier(classifier)
        label_encoder = LabelEncoder()
        label_encoder.fit(list(spec.label_classes))
        return FeaturePipeline(
            tfidf=TfidfFeaturizer(),
            categorical=CategoricalEncoder(
                categorical_fields=list(spec.categorical_fields),
                ordinal_fields=list(spec.ordinal_fields),
            ),
            numerical=NumericalScaler(
                scaled_fields=list(spec.scaled_numeric_fields),
                passthrough_fields=list(spec.passthrough_numeric_fields),
            ),
            entities=EntityExtractor(),
            label_encoder=label_encoder,
            spec=spec,
        )

    def _resolve_target_series(self, df: pd.DataFrame) -> pd.Series:
        candidates = (self.spec.target_column, *self.spec.fallback_target_columns)
        for candidate in candidates:
            if candidate in df.columns:
                return df[candidate].astype(str)
        raise KeyError(
            f"Missing target column for classifier '{self.spec.classifier}'. "
            f"Tried: {candidates}"
        )

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns: set[str] = set(self.spec.ordinal_fields)
        required_columns.update(self.spec.scaled_numeric_fields)
        required_columns.update(self.spec.passthrough_numeric_fields)
        if self.spec.include_temporal:
            required_columns.update({"hour_of_day", "day_of_week", "is_weekend", "is_after_hours"})

        missing = _missing_columns(df, required_columns)
        prepared = df
        if missing:
            prepared, _ = augment_inference_features_df_vectorized(df)

        if self.spec.requires_category_hint:
            prepared = _ensure_category_hint(prepared)
        return prepared

    def _build_text_columns(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        tfidf_text = _combine_text_fields(
            df,
            self.spec.text_tfidf_fields,
            separator=" ",
            placeholder="",
        )
        bert_text = _combine_text_fields(
            df,
            self.spec.text_bert_fields,
            separator=_SEP_TOKEN,
            placeholder="",
        )
        return tfidf_text, bert_text

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        prepared = self._prepare(train_df)
        y = self.label_encoder.transform(self._resolve_target_series(prepared))
        tfidf_text, _ = self._build_text_columns(prepared)
        self.tfidf.fit(tfidf_text, y)
        self.categorical.fit(prepared)
        self.numerical.fit(prepared)
        return self

    def transform(self, df: pd.DataFrame) -> dict[str, object]:
        prepared = self._prepare(df)
        tfidf_text, bert_text = self._build_text_columns(prepared)

        tfidf = self.tfidf.transform(tfidf_text)
        categorical = self.categorical.transform(prepared)
        numerical = self.numerical.transform(prepared)

        blocks = [categorical, numerical]
        if self.spec.include_entities:
            blocks.append(self.entities.transform(prepared))
        if self.spec.include_temporal:
            temporal = prepared[
                ["hour_of_day", "day_of_week", "is_weekend", "is_after_hours"]
            ].to_numpy(dtype=np.float32)
            blocks.append(temporal)

        tabular = np.hstack(blocks).astype(np.float32) if blocks else np.zeros((len(df), 0))
        texts = prepare_for_bert(bert_text)
        return {"tfidf": tfidf, "tabular": tabular, "text_bert": texts}

    def get_catboost_features(self, df: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray | None]:
        prepared = self._prepare(df)
        transformed = self.transform(df)
        tabular_sparse = sparse.csr_matrix(transformed["tabular"])
        X = sparse.hstack([transformed["tfidf"], tabular_sparse], format="csr")
        y = None
        if self.spec.target_column in df.columns or any(
            candidate in df.columns for candidate in self.spec.fallback_target_columns
        ):
            y = self.label_encoder.transform(self._resolve_target_series(prepared))
        return X, y

    def get_bert_features(self, df: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray | None]:
        transformed = self.transform(df)
        y = None
        if self.spec.target_column in df.columns or any(
            candidate in df.columns for candidate in self.spec.fallback_target_columns
        ):
            prepared = self._prepare(df)
            y = self.label_encoder.transform(self._resolve_target_series(prepared))
        return transformed["text_bert"], transformed["tabular"], y

    def decode_labels(self, label_ids: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(label_ids.astype(int))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "FeaturePipeline":
        return joblib.load(path)
