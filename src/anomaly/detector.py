"""Confidence anomaly detection against baseline statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src import config


@dataclass
class CategoryBaseline:
    """Historical confidence statistics for a category."""

    category: str
    confidence_mean: float
    confidence_std: float
    volume_count: int
    sample_count: int
    last_updated: datetime

    def confidence_zscore(self, confidence: float) -> float:
        if self.confidence_std < 1e-6:
            return 0.0
        return (confidence - self.confidence_mean) / self.confidence_std

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "confidence_mean": self.confidence_mean,
            "confidence_std": self.confidence_std,
            "volume_count": self.volume_count,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CategoryBaseline":
        return cls(
            category=str(payload["category"]),
            confidence_mean=float(payload["confidence_mean"]),
            confidence_std=float(payload["confidence_std"]),
            volume_count=int(payload["volume_count"]),
            sample_count=int(payload["sample_count"]),
            last_updated=datetime.fromisoformat(str(payload["last_updated"])),
        )


@dataclass
class AnomalyBaseline:
    """Complete anomaly baseline persisted to disk."""

    category_baselines: dict[str, CategoryBaseline] = field(default_factory=dict)
    overall_category_distribution: dict[str, float] = field(default_factory=dict)
    total_predictions: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "category_baselines": {
                category: stats.to_dict()
                for category, stats in self.category_baselines.items()
            },
            "overall_category_distribution": self.overall_category_distribution,
            "total_predictions": self.total_predictions,
            "created_at": self.created_at.isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AnomalyBaseline | None":
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                category_baselines={
                    category: CategoryBaseline.from_dict(stats)
                    for category, stats in payload["category_baselines"].items()
                },
                overall_category_distribution={
                    category: float(value)
                    for category, value in payload["overall_category_distribution"].items()
                },
                total_predictions=int(payload["total_predictions"]),
                created_at=datetime.fromisoformat(payload["created_at"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None


@dataclass(frozen=True)
class AnomalyResult:
    """Per-prediction anomaly output."""

    confidence_zscore: float
    is_confidence_anomaly: bool
    anomaly_reasons: list[str] = field(default_factory=list)


class AnomalyDetector:
    """Detects low-confidence outliers by category."""

    def __init__(
        self,
        baseline: AnomalyBaseline | None = None,
        zscore_threshold: float | None = None,
    ) -> None:
        self.baseline = baseline
        self.zscore_threshold = zscore_threshold or config.ANOMALY_CONFIDENCE_ZSCORE_THRESHOLD
        self.min_samples = config.ANOMALY_MIN_BASELINE_SAMPLES

    @classmethod
    def from_path(
        cls,
        path: Path = config.ANOMALY_BASELINE_PATH,
        zscore_threshold: float | None = None,
    ) -> "AnomalyDetector":
        return cls(
            baseline=AnomalyBaseline.load(path),
            zscore_threshold=zscore_threshold,
        )

    def analyze_prediction(
        self,
        predicted_category: str,
        category_confidence: float,
    ) -> AnomalyResult:
        if self.baseline is None:
            return AnomalyResult(confidence_zscore=0.0, is_confidence_anomaly=False)

        category_stats = self.baseline.category_baselines.get(predicted_category)
        if category_stats is None or category_stats.sample_count < self.min_samples:
            return AnomalyResult(confidence_zscore=0.0, is_confidence_anomaly=False)

        zscore = category_stats.confidence_zscore(category_confidence)
        is_anomaly = zscore < -self.zscore_threshold
        reasons: list[str] = []
        if is_anomaly:
            reasons.append(
                f"Confidence {category_confidence:.3f} is {abs(zscore):.1f}σ below "
                f"{predicted_category} baseline ({category_stats.confidence_mean:.3f})"
            )
        return AnomalyResult(
            confidence_zscore=zscore,
            is_confidence_anomaly=is_anomaly,
            anomaly_reasons=reasons,
        )
