"""Batch volume anomaly analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    epsilon = 1e-10
    p = (p + epsilon) / (p + epsilon).sum()
    q = (q + epsilon) / (q + epsilon).sum()
    m = (p + q) / 2.0
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


@dataclass(frozen=True)
class CategoryVolumeDetail:
    """Per-category volume comparison between baseline and current windows."""

    category: str
    baseline_count: int
    current_count: int
    baseline_pct: float
    current_pct: float
    pct_change: float


@dataclass
class VolumeAnomalyReport:
    """Volume anomaly analysis output."""

    analysis_timestamp: datetime
    baseline_total: int
    current_total: int
    volume_change_pct: float
    category_distribution_jsd: float
    is_volume_anomaly: bool
    is_distribution_anomaly: bool
    category_details: list[CategoryVolumeDetail] = field(default_factory=list)
    anomaly_summary: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "baseline_total": self.baseline_total,
            "current_total": self.current_total,
            "volume_change_pct": round(self.volume_change_pct, 2),
            "category_distribution_jsd": round(self.category_distribution_jsd, 4),
            "is_volume_anomaly": self.is_volume_anomaly,
            "is_distribution_anomaly": self.is_distribution_anomaly,
            "category_details": [
                {
                    **asdict(detail),
                    "baseline_pct": round(detail.baseline_pct, 4),
                    "current_pct": round(detail.current_pct, 4),
                    "pct_change": round(detail.pct_change, 2),
                }
                for detail in self.category_details
            ],
            "anomaly_summary": self.anomaly_summary,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def analyze_volume_patterns(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    category_col: str = "predicted_category",
    jsd_threshold: float | None = None,
    volume_change_threshold: float = 0.30,
    pct_point_change_threshold: float = 0.10,
) -> VolumeAnomalyReport:
    """Compare category distributions and ticket volume between windows."""
    if current_df.empty:
        raise ValueError("Predictions DataFrame is empty")
    if baseline_df.empty:
        raise ValueError("Baseline DataFrame is empty")

    threshold = config.ANOMALY_JSD_THRESHOLD if jsd_threshold is None else jsd_threshold
    categories = sorted(set(baseline_df[category_col].unique()) | set(current_df[category_col].unique()))

    baseline_counts = baseline_df[category_col].value_counts().reindex(categories, fill_value=0)
    current_counts = current_df[category_col].value_counts().reindex(categories, fill_value=0)

    baseline_total = int(len(baseline_df))
    current_total = int(len(current_df))
    baseline_denom = max(baseline_total, 1)
    current_denom = max(current_total, 1)

    baseline_dist = baseline_counts / baseline_denom
    current_dist = current_counts / current_denom
    distribution_jsd = js_divergence(baseline_dist.values, current_dist.values)
    volume_change = (current_total - baseline_total) / baseline_denom

    distribution_anomaly = distribution_jsd > threshold
    volume_anomaly = abs(volume_change) > volume_change_threshold

    details: list[CategoryVolumeDetail] = []
    summary: list[str] = []
    for category in categories:
        baseline_pct = float(baseline_dist[category])
        current_pct = float(current_dist[category])
        pct_change = current_pct - baseline_pct
        details.append(
            CategoryVolumeDetail(
                category=str(category),
                baseline_count=int(baseline_counts[category]),
                current_count=int(current_counts[category]),
                baseline_pct=baseline_pct,
                current_pct=current_pct,
                pct_change=pct_change,
            )
        )
        if abs(pct_change) > pct_point_change_threshold:
            direction = "increase" if pct_change > 0 else "decrease"
            summary.append(
                f"Category '{category}' {direction}: {abs(pct_change)*100:.1f}pp "
                f"({baseline_pct*100:.1f}% -> {current_pct*100:.1f}%)"
            )

    if distribution_anomaly:
        summary.append(f"Category distribution shift detected (JSD={distribution_jsd:.3f})")
    if volume_anomaly:
        direction = "increase" if volume_change > 0 else "decrease"
        summary.append(
            f"Total volume {direction}: {abs(volume_change):.1%} ({baseline_total}->{current_total})"
        )

    return VolumeAnomalyReport(
        analysis_timestamp=datetime.now(UTC),
        baseline_total=baseline_total,
        current_total=current_total,
        volume_change_pct=volume_change * 100,
        category_distribution_jsd=distribution_jsd,
        is_volume_anomaly=volume_anomaly,
        is_distribution_anomaly=distribution_anomaly,
        category_details=details,
        anomaly_summary=summary,
    )
