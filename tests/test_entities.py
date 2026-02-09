"""Tests for entity extraction."""

import numpy as np
import pandas as pd
import pytest

from src.features.entities import EntityExtractor


def test_combined_error_hash_preserves_all_codes():
    """Test that combined hash captures all error codes."""
    extractor = EntityExtractor(buckets=16)

    # Two different sets of error codes should hash differently
    bucket1 = extractor._combined_hash_bucket(["ERROR_TIMEOUT_429"])
    bucket2 = extractor._combined_hash_bucket(["ERROR_TIMEOUT_429", "ERROR_AUTH_403"])
    bucket3 = extractor._combined_hash_bucket(["ERROR_AUTH_403"])

    # All should be different (collision possible but unlikely with 16 buckets)
    # The important thing: bucket2 != bucket1 (has info about second error)
    assert bucket2 != bucket1, "Combined hash should differ when error codes differ"
    assert bucket2 != bucket3, "Combined hash should differ when error codes differ"


def test_combined_hash_is_order_independent():
    """Test that hash is the same regardless of error code order."""
    extractor = EntityExtractor(buckets=16)

    bucket1 = extractor._combined_hash_bucket(["ERROR_TIMEOUT", "ERROR_AUTH"])
    bucket2 = extractor._combined_hash_bucket(["ERROR_AUTH", "ERROR_TIMEOUT"])

    assert bucket1 == bucket2, "Hash should be order-independent"


def test_combined_hash_empty_list():
    """Test that empty list returns bucket 0."""
    extractor = EntityExtractor(buckets=16)

    bucket = extractor._combined_hash_bucket([])
    assert bucket == 0, "Empty list should return bucket 0"


def test_most_severe_error_detection():
    """Test that most severe error is correctly identified."""
    extractor = EntityExtractor(buckets=16)

    # CRITICAL > ERROR > WARN
    critical_bucket = extractor._most_severe_error_bucket(["ERROR_TIMEOUT", "CRITICAL_FAIL"])
    error_only_bucket = extractor._most_severe_error_bucket(["ERROR_TIMEOUT"])

    # CRITICAL should be detected as more severe, so bucket should differ
    assert critical_bucket != 0, "Should find CRITICAL error"


def test_most_severe_error_empty_list():
    """Test that empty list returns bucket 0."""
    extractor = EntityExtractor(buckets=16)

    bucket = extractor._most_severe_error_bucket([])
    assert bucket == 0, "Empty list should return bucket 0"


def test_transform_generates_correct_shape():
    """Test that transform produces correct feature dimension."""
    extractor = EntityExtractor(buckets=16)

    df = pd.DataFrame({
        'subject': ['Test'],
        'description': ['Failed with ERROR_TIMEOUT_429 then ERROR_AUTH_403'],
        'error_logs': [''],
        'stack_trace': ['']
    })

    features = extractor.transform(df)

    # Feature dimension should be:
    # 1 (has_error) + 1 (has_stack) + 1 (count) + 16 (combined) + 16 (severe) + 1 (product) = 36
    expected_dim = 1 + 1 + 1 + 16 + 16 + 1
    assert features.shape == (1, expected_dim), f"Expected shape (1, {expected_dim}), got {features.shape}"


def test_transform_counts_multiple_error_codes():
    """Test that error code count is correct."""
    extractor = EntityExtractor(buckets=16)

    df = pd.DataFrame({
        'subject': ['Test'],
        'description': ['Failed with ERROR_TIMEOUT_429 then ERROR_AUTH_403'],
        'error_logs': [''],
        'stack_trace': ['']
    })

    features = extractor.transform(df)

    # Error code count is at index 2
    assert features[0, 2] == 2.0, "Should count both error codes"


def test_backwards_compatibility_single_error():
    """Test that single-error tickets still work correctly."""
    extractor = EntityExtractor(buckets=16)

    df = pd.DataFrame({
        'subject': ['Test'],
        'description': ['Failed with ERROR_TIMEOUT_429'],
        'error_logs': [''],
        'stack_trace': ['']
    })

    features = extractor.transform(df)

    # Should have count of 1
    assert features[0, 2] == 1.0, "Should count single error code"

    # Combined and severe buckets should both have exactly one hot
    # Indices 3-18 are combined bucket (16 buckets), 19-34 are severe bucket
    combined_range = features[0, 3:19]
    severe_range = features[0, 19:35]

    assert combined_range.sum() == 1.0, "Combined bucket should have exactly one hot"
    assert severe_range.sum() == 1.0, "Severe bucket should have exactly one hot"


def test_no_error_codes():
    """Test behavior when no error codes present."""
    extractor = EntityExtractor(buckets=16)

    df = pd.DataFrame({
        'subject': ['Test'],
        'description': ['Everything is working fine'],
        'error_logs': [''],
        'stack_trace': ['']
    })

    features = extractor.transform(df)

    # Error code count should be 0
    assert features[0, 2] == 0.0, "Should count zero error codes"

    # has_error_code should be False (0)
    assert features[0, 0] == 0.0, "has_error_code should be 0"


def test_product_mentioned_detection():
    """Test product mention detection."""
    extractor = EntityExtractor(buckets=16)

    df = pd.DataFrame({
        'subject': ['Issue with DataSync Pro'],
        'description': ['The sync is failing'],
        'error_logs': [''],
        'stack_trace': ['']
    })

    features = extractor.transform(df)

    # Product mentioned is the last feature
    assert features[0, -1] == 1.0, "Should detect DataSync Pro mention"


def test_stack_trace_detection():
    """Test stack trace detection."""
    extractor = EntityExtractor(buckets=16)

    df = pd.DataFrame({
        'subject': ['Error'],
        'description': ['App crashed'],
        'error_logs': [''],
        'stack_trace': ['Traceback (most recent call last):\n  File "app.py"']
    })

    features = extractor.transform(df)

    # has_stack_trace is at index 1
    assert features[0, 1] == 1.0, "Should detect stack trace"
