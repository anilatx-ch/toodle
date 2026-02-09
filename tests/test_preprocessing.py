"""Tests for preprocessing functions."""

import pandas as pd
import pytest

from src.features.preprocessing import (
    augment_inference_features,
    augment_inference_features_df_vectorized,
    detect_leakage_warning,
    PRIORITY_MAP,
)
from src.api.schemas import TicketInput


def test_vectorized_produces_temporal_features():
    df = pd.DataFrame({
        'ticket_id': ['T1'],
        'subject': ['Test'],
        'description': ['Test description'],
        'created_at': ['2024-01-15T10:30:00Z'],
        'severity': ['P2'],
    })

    result, _ = augment_inference_features_df_vectorized(df)

    assert 'hour_of_day' in result.columns
    assert 'day_of_week' in result.columns
    assert 'is_weekend' in result.columns
    assert 'is_after_hours' in result.columns
    assert result.iloc[0]['hour_of_day'] == 10


def test_vectorized_produces_text_features():
    df = pd.DataFrame({
        'ticket_id': ['T1'],
        'subject': ['Login failed'],
        'description': ['Cannot access app'],
        'error_logs': ['ERROR_AUTH_401'],
        'stack_trace': [''],
        'created_at': ['2024-01-15T10:30:00Z'],
        'severity': ['P2'],
    })

    result, _ = augment_inference_features_df_vectorized(df)

    assert 'text_combined_tfidf' in result.columns
    assert 'text_combined_bert' in result.columns
    assert '[SEP]' in result.iloc[0]['text_combined_bert']


def test_vectorized_produces_derived_features():
    df = pd.DataFrame({
        'ticket_id': ['T1'],
        'subject': ['Error'],
        'description': ['Got ERROR_TIMEOUT_429'],
        'error_logs': [''],
        'stack_trace': ['Traceback...'],
        'created_at': ['2024-01-15T10:30:00Z'],
        'severity': ['P2'],
    })

    result, _ = augment_inference_features_df_vectorized(df)

    assert 'has_error_code' in result.columns
    assert 'has_stack_trace' in result.columns
    assert 'ticket_text_length' in result.columns
    assert result.iloc[0]['has_error_code'] == True
    assert result.iloc[0]['has_stack_trace'] == True


def test_vectorized_produces_ordinal_features():
    df = pd.DataFrame({
        'ticket_id': ['T1'],
        'subject': ['Test'],
        'description': ['Test'],
        'priority': ['high'],
        'severity': ['P1'],
        'created_at': ['2024-01-15T10:30:00Z'],
    })

    result, _ = augment_inference_features_df_vectorized(df)

    assert 'priority_ordinal' in result.columns
    assert 'severity_ordinal' in result.columns
    assert result.iloc[0]['priority_ordinal'] == PRIORITY_MAP['high']


def test_vectorized_detects_leakage():
    df = pd.DataFrame({
        'ticket_id': ['T1', 'T2'],
        'subject': ['Issue', 'Bug'],
        'description': ['Root cause identified as network issue', 'Normal description'],
        'created_at': ['2024-01-15T10:30:00Z', '2024-01-15T11:00:00Z'],
        'severity': ['P2', 'P3'],
    })

    result, warnings = augment_inference_features_df_vectorized(df)

    assert warnings[0] == 'possible_leakage_pattern'
    assert warnings[1] is None


def test_vectorized_handles_missing_priority():
    df = pd.DataFrame({
        'ticket_id': ['T1'],
        'subject': ['Test'],
        'description': ['Test'],
        'severity': ['P2'],
        'created_at': ['2024-01-15T10:30:00Z'],
    })

    result, _ = augment_inference_features_df_vectorized(df)

    assert 'priority_ordinal' in result.columns
    assert result.iloc[0]['priority_ordinal'] == -1


def test_detect_leakage_warning():
    ticket_data = {
        'subject': 'Issue',
        'description': 'Root cause identified as database timeout',
        'error_logs': '',
        'stack_trace': ''
    }

    warning = detect_leakage_warning(ticket_data)
    assert warning == 'possible_leakage_pattern'

    clean_ticket = {
        'subject': 'Issue',
        'description': 'Database timeout occurred',
        'error_logs': '',
        'stack_trace': ''
    }

    warning = detect_leakage_warning(clean_ticket)
    assert warning is None
