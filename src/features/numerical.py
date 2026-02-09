"""Numerical feature scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class NumericalScaler:
    scaled_fields: list[str] = None
    passthrough_fields: list[str] = None

    def __post_init__(self) -> None:
        # If None (not specified), use defaults. If explicitly empty list, respect it.
        if self.scaled_fields is None:
            self.scaled_fields = [
                "account_age_days",
                "account_monthly_value",
                "product_version_age_days",
                "ticket_text_length",
            ]
        if self.passthrough_fields is None:
            self.passthrough_fields = ["previous_tickets", "attachments_count"]
        self.scaler = StandardScaler()
        self.is_fitted = False

    @staticmethod
    def _with_default_columns(
        df: pd.DataFrame, columns: list[str], default_value: float | int
    ) -> pd.DataFrame:
        if not columns:
            return pd.DataFrame(index=df.index)
        values = {column: df[column] if column in df.columns else default_value for column in columns}
        return pd.DataFrame(values, index=df.index)

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        values = self._with_default_columns(df, self.scaled_fields, 0.0)
        if "account_monthly_value" in values.columns:
            values["account_monthly_value"] = np.log1p(values["account_monthly_value"].clip(lower=0))
        return values

    def fit(self, df: pd.DataFrame) -> "NumericalScaler":
        scaled = self._prep(df)
        if not scaled.empty:
            self.scaler.fit(scaled)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("NumericalScaler must be fitted before transform")
        scaled_source = self._prep(df)
        if scaled_source.empty:
            scaled = np.zeros((len(df), 0), dtype=np.float32)
        else:
            scaled = self.scaler.transform(scaled_source)
        passthrough_source = self._with_default_columns(df, self.passthrough_fields, 0.0)
        passthrough = passthrough_source.to_numpy(dtype=np.float32)
        return np.hstack([scaled, passthrough]).astype(np.float32)
