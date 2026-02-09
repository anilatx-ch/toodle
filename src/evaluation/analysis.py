"""Error analysis utilities."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd


def generate_error_analysis(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    n_samples: int = 50,
) -> list[dict[str, Any]]:
    """Generate detailed error analysis for misclassified samples."""
    errors = np.where(y_true != y_pred)[0].tolist()
    random.Random(42).shuffle(errors)
    selected = errors[:n_samples]

    out = []
    for idx in selected:
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        probs = y_proba[idx]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [[class_names[i], float(probs[i])] for i in top3_idx]

        row = df_test.iloc[idx]
        out.append(
            {
                "ticket_id": str(row["ticket_id"]),
                "true_category": true_label,
                "predicted_category": pred_label,
                "confidence": float(np.max(probs)),
                "subject": str(row.get("subject", "")),
                "description_snippet": str(row.get("description", ""))[:200],
                "top_3_predictions": top3,
            }
        )
    return out


def generate_confusion_clusters(
    cm: np.ndarray, class_names: list[str], top_k: int = 5
) -> list[dict[str, Any]]:
    """Identify most common confusion patterns."""
    clusters = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                clusters.append(
                    {
                        "true": class_names[i],
                        "pred": class_names[j],
                        "count": count,
                    }
                )
    clusters = sorted(clusters, key=lambda x: x["count"], reverse=True)
    return clusters[:top_k]


def confidence_analysis(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Analyze prediction confidence patterns."""
    confidence = np.max(y_proba, axis=1)
    is_error = y_true != y_pred

    high_conf_errors = np.sum(is_error & (confidence > 0.8))
    low_conf_errors = np.sum(is_error & (confidence < 0.5))
    total_errors = np.sum(is_error)

    return {
        "total_errors": int(total_errors),
        "high_confidence_errors": int(high_conf_errors),
        "low_confidence_errors": int(low_conf_errors),
        "high_confidence_error_rate": float(high_conf_errors / max(total_errors, 1)),
        "low_confidence_error_rate": float(low_conf_errors / max(total_errors, 1)),
    }
