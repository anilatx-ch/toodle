"""Shared pytest fixtures for TOODLE tests."""

import pandas as pd
import pytest


@pytest.fixture
def feature_df():
    """Minimal DataFrame for feature pipeline testing."""
    return pd.DataFrame({
        'ticket_id': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'subject': ['Bug in login', 'Billing question', 'API timeout', 'Feature request: export', 'Security concern'],
        'description': ['Cannot login to app', 'Invoice is wrong', 'Timeout after 30s', 'Need CSV export', 'Data leak suspected'],
        'error_logs': ['', 'ERROR_BILLING_500', '', '', ''],
        'stack_trace': ['', '', 'Traceback...', '', ''],
        'created_at': ['2024-01-15T10:30:00Z', '2024-01-15T14:00:00Z', '2024-01-16T09:00:00Z', '2024-01-16T19:00:00Z', '2024-01-17T20:00:00Z'],
        'category': ['Technical Issue', 'Account Management', 'Technical Issue', 'Feature Request', 'Security'],
        'split': ['train', 'train', 'train', 'val', 'test'],
    })
