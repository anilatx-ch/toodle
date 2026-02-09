"""Text feature processing for TF-IDF and BERT input preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from src import config


@dataclass
class TfidfFeaturizer:
    max_features: int = config.TFIDF_MAX_FEATURES
    k_best: int = config.CHI2_SELECT_K

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2))
        self.selector = SelectKBest(score_func=chi2, k=self.k_best)
        self.is_fitted = False

    def fit(self, train_texts: pd.Series, train_labels: np.ndarray) -> "TfidfFeaturizer":
        X = self.vectorizer.fit_transform(train_texts.fillna(""))
        self.selector.fit(X, train_labels)
        self.is_fitted = True
        return self

    def transform(self, texts: pd.Series) -> sparse.csr_matrix:
        if not self.is_fitted:
            raise RuntimeError("TfidfFeaturizer must be fitted before transform")
        X = self.vectorizer.transform(texts.fillna(""))
        X = self.selector.transform(X)
        return X.tocsr()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TfidfFeaturizer":
        return joblib.load(path)


def prepare_for_bert(text_series: pd.Series) -> list[str]:
    """Prepare pre-concatenated BERT strings."""
    return text_series.fillna("").astype(str).tolist()
