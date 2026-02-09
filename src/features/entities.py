"""Regex/entity extraction features."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


ERROR_RE = re.compile(r"ERROR_\w+|\w+_\d{3}")
PRODUCTS = [
    "DataSync Pro",
    "Analytics Dashboard",
    "API Gateway",
    "StreamProcessor",
    "CloudBackup Enterprise",
]

# Severity order for error code prioritization
SEVERITY_ORDER = {
    'CRITICAL': 1000,
    'FATAL': 900,
    'ERROR': 800,
    'WARN': 700,
    'WARNING': 700,
    'INFO': 600,
}


@dataclass
class EntityExtractor:
    buckets: int = config.ERROR_CODE_HASH_BUCKETS

    def _stable_bucket(self, token: str) -> int:
        """Hash a single token into a stable bucket."""
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.buckets

    def _combined_hash_bucket(self, tokens: list[str]) -> int:
        """
        Hash multiple tokens into a single stable bucket.

        This preserves information about ALL error codes while keeping
        fixed feature dimension. Tokens are sorted for consistency.

        Examples:
            ["ERROR_TIMEOUT"] → bucket_42
            ["ERROR_AUTH", "ERROR_TIMEOUT"] → bucket_17 (combined signature)
            ["ERROR_TIMEOUT", "ERROR_AUTH"] → bucket_17 (order-independent)

        Args:
            tokens: List of error code strings

        Returns:
            Bucket index (0 to self.buckets-1)
        """
        if not tokens:
            return 0  # Special bucket for "no error codes"

        # Sort for order-independence
        sorted_tokens = sorted(tokens)
        combined = ",".join(sorted_tokens)
        digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.buckets

    def _most_severe_error_bucket(self, tokens: list[str]) -> int:
        """
        Get bucket for the most severe error code.

        Severity heuristic:
        1. CRITICAL_XXX > FATAL_XXX > ERROR_XXX > WARN_XXX > INFO_XXX
        2. Among same prefix, higher numeric = more severe (e.g., 503 > 404)

        Args:
            tokens: List of error code strings

        Returns:
            Bucket index for most severe error
        """
        if not tokens:
            return 0

        def severity_score(token: str) -> tuple[int, int]:
            """Return (prefix_severity, numeric_severity) for sorting."""
            for prefix, score in SEVERITY_ORDER.items():
                if token.startswith(prefix):
                    # Extract numeric part if present (e.g., 404 from ERROR_HTTP_404)
                    numeric_part = 0
                    for part in token.split('_'):
                        if part.isdigit():
                            numeric_part = int(part)
                    return (score, numeric_part)
            return (0, 0)  # Unknown severity

        # Sort by severity descending, get most severe
        most_severe = max(tokens, key=severity_score)
        return self._stable_bucket(most_severe)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract entity features from ticket DataFrame.

        Features:
        - has_error_code: Binary, any error pattern found
        - has_stack_trace: Binary, stack trace present
        - error_code_count: Float, number of error codes found
        - combined_error_bucket: Categorical, hash of ALL error codes (NEW)
        - most_severe_error_bucket: Categorical, most severe error code (NEW)
        - product_mentioned: Binary, product name in text
        """
        subject = df["subject"].fillna("")
        description = df["description"].fillna("")
        error_logs = df["error_logs"].fillna("")
        stack_trace = df["stack_trace"].fillna("")

        has_error_code = df.get("has_error_code", False)
        if not isinstance(has_error_code, pd.Series):
            has_error_code = (description + " " + error_logs).str.contains(ERROR_RE, regex=True)

        has_stack_trace = df.get("has_stack_trace", False)
        if not isinstance(has_stack_trace, pd.Series):
            has_stack_trace = stack_trace.str.len() > 0

        combo = (description + " " + error_logs).astype(str)

        # Vectorized error code extraction
        all_error_codes = combo.str.findall(ERROR_RE)

        # Error code count
        error_code_count = np.array(
            [float(len(codes)) for codes in all_error_codes],
            dtype=np.float32
        ).reshape(-1, 1)

        # Combined hash of ALL error codes (preserves multi-error information)
        combined_bucket_one_hot = np.zeros((len(df), self.buckets), dtype=np.float32)
        for i, codes in enumerate(all_error_codes):
            bucket = self._combined_hash_bucket(codes)
            combined_bucket_one_hot[i, bucket] = 1.0

        # Most severe error bucket
        severe_bucket_one_hot = np.zeros((len(df), self.buckets), dtype=np.float32)
        for i, codes in enumerate(all_error_codes):
            bucket = self._most_severe_error_bucket(codes)
            severe_bucket_one_hot[i, bucket] = 1.0

        # Product mention (vectorized)
        product_mentioned = np.zeros((len(df), 1), dtype=np.float32)
        merged_text = (subject + " " + description).str.lower()
        for product in PRODUCTS:
            product_lower = product.lower()
            contains_product = merged_text.str.contains(product_lower, regex=False).to_numpy().astype(np.float32)
            product_mentioned[:, 0] = np.maximum(product_mentioned[:, 0], contains_product)

        # Stack all features
        # REPLACED: first_match_bucket (OLD, lossy)
        # WITH: combined_bucket_one_hot, severe_bucket_one_hot (NEW, lossless)
        return np.hstack(
            [
                has_error_code.astype(np.float32).to_numpy().reshape(-1, 1),
                has_stack_trace.astype(np.float32).to_numpy().reshape(-1, 1),
                error_code_count,
                combined_bucket_one_hot,    # All error codes hashed together
                severe_bucket_one_hot,      # Most severe error
                product_mentioned,
            ]
        ).astype(np.float32)
