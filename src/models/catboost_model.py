"""CatBoost classifier wrapper for ticket category prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

try:  # pragma: no cover - runtime dependency
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - runtime dependency
    CatBoostClassifier = None


class CatBoostTicketClassifier:
    """Thin wrapper around CatBoost with a stable project interface."""

    def __init__(
        self,
        *,
        params: dict[str, object] | None = None,
        label_classes: list[str] | None = None,
    ) -> None:
        self.params = {**config.CATBOOST_BASELINE_PARAMS, **(params or {})}
        self.label_classes = list(label_classes or config.CATEGORY_CLASSES)
        self.model: CatBoostClassifier | None = None

    def fit(
        self,
        X_train,
        y_train: np.ndarray,
        X_val=None,
        y_val: np.ndarray | None = None,
    ) -> "CatBoostTicketClassifier":
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoost is not installed")

        self.model = CatBoostClassifier(**self.params)
        fit_kwargs: dict[str, object] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)
            fit_kwargs["use_best_model"] = False

        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def _require_model(self) -> CatBoostClassifier:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return self.model

    def predict_proba(self, X) -> np.ndarray:
        model = self._require_model()
        return np.asarray(model.predict_proba(X), dtype=np.float32)

    def predict_label_ids(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).astype(np.int64)

    def predict_labels(self, X) -> np.ndarray:
        label_ids = self.predict_label_ids(X)
        labels = np.asarray(self.label_classes, dtype=object)
        return labels[label_ids]

    def get_feature_importance(self) -> pd.DataFrame:
        model = self._require_model()
        scores = np.asarray(model.get_feature_importance(), dtype=np.float32)
        rows = [
            {"feature": f"f{idx}", "importance": float(score)}
            for idx, score in enumerate(scores)
        ]
        return pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)

    def save(self, path: str | Path) -> None:
        model = self._require_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))

        metadata = {
            "label_classes": self.label_classes,
            "params": self.params,
        }
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostTicketClassifier":
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoost is not installed")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CatBoost model not found: {path}")

        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        metadata: dict[str, object] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        model = cls(
            params=metadata.get("params") if isinstance(metadata.get("params"), dict) else None,
            label_classes=(
                metadata.get("label_classes")
                if isinstance(metadata.get("label_classes"), list)
                else None
            ),
        )
        model.model = CatBoostClassifier()
        model.model.load_model(str(path))
        return model
