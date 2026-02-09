"""Shared preprocessing logic for raw ticket data."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from src.api.schemas import TicketInput, TicketInputMultiClassifier
from src.features.entities import ERROR_RE

LEAKAGE_RE = re.compile(r"root cause identified as", flags=re.IGNORECASE)

PRIORITY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SEVERITY_MAP = {"P4": 0, "P3": 1, "P2": 2, "P1": 3, "P0": 4}


def _parse_created_at(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def detect_leakage_warning(ticket_data: Mapping[str, object]) -> Optional[str]:
    subject = str(ticket_data.get("subject") or "")
    description = str(ticket_data.get("description") or "")
    error_logs = str(ticket_data.get("error_logs") or "")
    stack_trace = str(ticket_data.get("stack_trace") or "")
    merged_text = f"{subject} {description} {error_logs} {stack_trace}".strip()
    return "possible_leakage_pattern" if LEAKAGE_RE.search(merged_text) else None


def augment_inference_features(
    ticket: TicketInput | TicketInputMultiClassifier,
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Transform a raw TicketInput object into a single-row DataFrame compatible with FeaturePipeline.

    This function encapsulates the logic to derive features like 'hour_of_day', 'text_combined_bert',
    and 'has_error_code' from the raw input fields.
    """
    created = _parse_created_at(ticket.created_at)

    error_logs = ticket.error_logs or ""
    stack_trace = ticket.stack_trace or ""
    merged_text = f"{ticket.subject} {ticket.description} {error_logs} {stack_trace}".strip()

    warning = detect_leakage_warning(ticket.model_dump())

    priority_value = getattr(ticket, "priority", None)
    priority_ordinal = PRIORITY_MAP.get(priority_value) if priority_value is not None else -1

    row = {
        "ticket_id": ticket.ticket_id,
        "created_at": ticket.created_at,
        "product": ticket.product,
        "product_module": ticket.product_module,
        "channel": ticket.channel,
        "customer_tier": ticket.customer_tier,
        "environment": ticket.environment,
        "language": ticket.language,
        "region": ticket.region,
        "priority": priority_value,
        "severity": ticket.severity,
        "priority_ordinal": priority_ordinal,
        "severity_ordinal": SEVERITY_MAP.get(ticket.severity),
        "subject": ticket.subject,
        "description": ticket.description,
        "error_logs": ticket.error_logs,
        "stack_trace": ticket.stack_trace,
        "account_age_days": ticket.account_age_days,
        "account_monthly_value": float(ticket.account_monthly_value),
        "previous_tickets": int(ticket.previous_tickets),
        "product_version_age_days": int(ticket.product_version_age_days),
        "attachments_count": int(ticket.attachments_count),
        "ticket_text_length": len(merged_text),
        "has_error_code": bool(ERROR_RE.search(f"{ticket.description} {error_logs}")),
        "has_stack_trace": len(stack_trace.strip()) > 0,
        "hour_of_day": created.hour,
        "day_of_week": (created.weekday() + 1) % 7,
        "is_weekend": int(created.weekday() >= 5),
        "is_after_hours": int(created.hour < 8 or created.hour >= 18),
        "text_combined_tfidf": f"{ticket.subject} {ticket.description} "
        f"{error_logs if error_logs else '[NO_ERROR_LOG]'} "
        f"{stack_trace if stack_trace else '[NO_STACK_TRACE]'}".strip(),
        "text_combined_bert": (
            f"{ticket.subject} [SEP] {ticket.description}"
            + (f" [SEP] {error_logs}" if error_logs else "")
            + (f" [SEP] {stack_trace}" if stack_trace else "")
        ),
        "category_hint": "",
    }

    if row["severity_ordinal"] is None:
        raise ValueError("Invalid severity value")
    if priority_value is not None and row["priority_ordinal"] is None:
        raise ValueError("Invalid priority value")

    return pd.DataFrame([row]), warning


def augment_inference_features_df_vectorized(
    df: pd.DataFrame,
    category_hint_column: str = "predicted_category"
) -> tuple[pd.DataFrame, list[Optional[str]]]:
    """
    Vectorized version of augment_inference_features for batch processing.

    This is ~100x faster than iterrows() for large datasets.
    Skips Pydantic validation since data is already validated at source (DB/API).

    Args:
        df: Source DataFrame with raw ticket columns
        category_hint_column: Column name to use for category_hint (default: predicted_category)

    Returns:
        tuple of (augmented_df, warnings_list)
        - augmented_df: DataFrame with all derived features added
        - warnings_list: List of warning strings (one per row)
    """
    result_df = df.copy()

    # Temporal features
    if "created_at" in result_df.columns:
        created_at = result_df["created_at"]
        created = pd.to_datetime(created_at.str.replace("Z", "+00:00", regex=False) if pd.api.types.is_string_dtype(created_at) else created_at, utc=True)

        result_df["hour_of_day"] = created.dt.hour
        result_df["day_of_week"] = (created.dt.dayofweek + 1) % 7
        result_df["is_weekend"] = (created.dt.dayofweek >= 5).astype(int)
        result_df["is_after_hours"] = ((created.dt.hour < 8) | (created.dt.hour >= 18)).astype(int)

    # Text features
    subject = result_df["subject"].fillna("") if "subject" in result_df.columns else pd.Series([""] * len(result_df))
    description = result_df["description"].fillna("") if "description" in result_df.columns else pd.Series([""] * len(result_df))
    error_logs = result_df["error_logs"].fillna("") if "error_logs" in result_df.columns else pd.Series([""] * len(result_df))
    stack_trace = result_df["stack_trace"].fillna("") if "stack_trace" in result_df.columns else pd.Series([""] * len(result_df))

    # Combined text for TF-IDF
    error_logs_tfidf = error_logs.where(error_logs != "", "[NO_ERROR_LOG]")
    stack_trace_tfidf = stack_trace.where(stack_trace != "", "[NO_STACK_TRACE]")
    result_df["text_combined_tfidf"] = (
        subject + " " + description + " " + error_logs_tfidf + " " + stack_trace_tfidf
    ).str.strip()

    # Combined text for BERT (with dynamic [SEP])
    def build_bert_text_vectorized(row):
        parts = [str(row.get("subject", "")), "[SEP]", str(row.get("description", ""))]
        el = row.get("error_logs")
        st = row.get("stack_trace")
        if pd.notna(el) and str(el).strip():
            parts.extend(["[SEP]", str(el)])
        if pd.notna(st) and str(st).strip():
            parts.extend(["[SEP]", str(st)])
        return " ".join(parts)

    result_df["text_combined_bert"] = result_df.apply(build_bert_text_vectorized, axis=1)

    # Derived features
    merged_for_error_check = subject + " " + description + " " + error_logs
    result_df["has_error_code"] = merged_for_error_check.str.contains(ERROR_RE, regex=True)
    result_df["has_stack_trace"] = stack_trace.str.len() > 0

    # Ticket text length
    merged_text = subject + " " + description + " " + error_logs + " " + stack_trace
    result_df["ticket_text_length"] = merged_text.str.len()

    # Priority/Severity ordinal
    if "priority" in result_df.columns:
        result_df["priority_ordinal"] = result_df["priority"].map(PRIORITY_MAP).fillna(-1).astype(int)
    else:
        result_df["priority_ordinal"] = -1

    if "severity" in result_df.columns:
        result_df["severity_ordinal"] = result_df["severity"].map(SEVERITY_MAP)
        if result_df["severity_ordinal"].isna().any():
            invalid_severities = result_df[result_df["severity_ordinal"].isna()]["severity"].unique()
            raise ValueError(f"Invalid severity value(s) found in data: {invalid_severities}")
    else:
        result_df["severity_ordinal"] = -1

    # Category hint
    if category_hint_column in result_df.columns:
        category_hint = result_df[category_hint_column].fillna("")
    elif "category" in result_df.columns:
        category_hint = result_df["category"].fillna("")
    else:
        category_hint = pd.Series([""] * len(result_df), index=result_df.index)

    result_df["category_hint"] = category_hint.astype(str)

    # Leakage warning detection
    merged_for_leakage = (subject + " " + description + " " + error_logs + " " + stack_trace).str.lower()
    has_leakage = merged_for_leakage.str.contains(LEAKAGE_RE, regex=True)
    warnings_list = [
        "possible_leakage_pattern" if v else None
        for v in has_leakage.tolist()
    ]

    return result_df, warnings_list
