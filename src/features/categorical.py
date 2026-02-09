"""Categorical and ordinal feature handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src import config


@dataclass
class CategoricalEncoder:
    categorical_fields: list[str] = None
    ordinal_fields: list[str] = None

    def __post_init__(self) -> None:
        # If None (not specified), use defaults. If explicitly empty list, respect it.
        if self.categorical_fields is None:
            self.categorical_fields = config.CREATION_TIME_CATEGORICAL_FIELDS
        if self.ordinal_fields is None:
            self.ordinal_fields = config.CREATION_TIME_ORDINAL_FIELDS
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.is_fitted = False
        self._has_categorical = bool(self.categorical_fields)

    @staticmethod
    def _with_default_columns(
        df: pd.DataFrame, columns: list[str], default_value: str | float | int
    ) -> pd.DataFrame:
        if not columns:
            return pd.DataFrame(index=df.index)
        values = {column: df[column] if column in df.columns else default_value for column in columns}
        return pd.DataFrame(values, index=df.index)

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        if self._has_categorical:
            source = self._with_default_columns(df, self.categorical_fields, "__missing__")
            self.encoder.fit(source)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("CategoricalEncoder must be fitted before transform")
        if self._has_categorical:
            source = self._with_default_columns(df, self.categorical_fields, "__missing__")
            one_hot = self.encoder.transform(source)
        else:
            one_hot = np.zeros((len(df), 0), dtype=np.float32)
        ordinal_source = self._with_default_columns(df, self.ordinal_fields, 0)
        ordinal = ordinal_source.to_numpy(dtype=np.float32)
        return np.hstack([one_hot, ordinal]).astype(np.float32)
