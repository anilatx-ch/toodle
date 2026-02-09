"""Data encoding utilities for subcategory independence investigation.

This module provides functions to prepare data for statistical testing by:
1. Removing timestamps from error logs (reduces cardinality)
2. Extracting error codes from descriptions
3. Masking error codes with placeholders (enables pattern generalization)
4. Encoding categorical features as integers (required for chi-squared tests)

These transformations were used to create data_int_encoded.pkl for the
subcategory independence investigation. Consolidated here
for reproducibility and documentation purposes.
"""

import re
import numpy as np
import pandas as pd


ERR_REGEX = r'ERROR_[A-Z0-9_]+'


def remove_timestamps_from_logs(error_logs: str) -> str:
    """Remove timestamps (first 20 chars) from each line of error_logs.

    Purpose: Reduce cardinality of error_logs feature by normalizing
    timestamp variations.

    Args:
        error_logs: Multi-line string with timestamp prefix on each line

    Returns:
        error_logs with timestamps stripped from each line
    """
    if not error_logs or pd.isna(error_logs):
        return error_logs

    lines = error_logs.split('\n')
    cleaned_lines = []
    for line in lines:
        if line and len(line) > 20:
            cleaned_lines.append(line[20:])
        elif line:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def process_description(desc: str) -> tuple[str | None, str | None]:
    """Extract error code and create cleaned description.

    Purpose: Separate error code signal from description text, enabling
    both template-based analysis and error code pattern matching.

    Args:
        desc: Description text potentially containing error codes

    Returns:
        Tuple of (error_code, cleaned_description) where error_code is
        the first ERROR_* pattern found and cleaned_description has all
        ERROR_* patterns replaced with <ERR_CODE> placeholder
    """
    if not desc:
        return None, None

    # Find the first match for the error code
    match = re.search(ERR_REGEX, desc)
    err_code = match.group(0) if match else None

    # Replace all error codes with placeholder
    cleaned_desc = re.sub(ERR_REGEX, '<ERR_CODE>', desc)

    return err_code, cleaned_desc


def mask_subject(subject: str, err_code: str | None) -> str:
    """Replace error code in subject with placeholder.

    Purpose: Create generalized subject templates that match across
    different error codes, enabling pattern-based analysis.

    Args:
        subject: Subject line text
        err_code: Error code to replace (from process_description)

    Returns:
        Subject with error code replaced by <ERR_CODE>, or original if
        no error code present
    """
    if not subject:
        return ""

    # If there's an error code, replace it; otherwise return original
    if err_code and pd.notna(err_code):
        return subject.replace(err_code, '<ERR_CODE>')
    return subject


def encode_column_to_int(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Encode categorical column to integers, preserving NaN.

    Purpose: Chi-squared and mutual information tests require integer-
    encoded categorical features. This encoding maintains NaN as -1
    sentinel value.

    Args:
        series: Pandas Series with categorical or string values

    Returns:
        Tuple of (codes, uniques) where:
        - codes: Integer array (NaN encoded as -1)
        - uniques: Array of unique non-NaN values in encoding order
    """
    # For NaN values, use -1 to indicate missing
    # For non-NaN values, use factorize to get integer codes
    mask = series.isna()
    codes, uniques = pd.factorize(series[~mask])
    full_codes = np.full(len(series), -1, dtype=int)
    full_codes[~mask] = codes
    return full_codes, uniques


def prepare_data_for_testing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all encoding transformations to raw ticket data.

    Creates integer-encoded features suitable for statistical testing:
    - Removes timestamps from error_logs
    - Extracts error codes from descriptions
    - Masks error codes in subjects
    - Encodes all features as integers

    Args:
        df: DataFrame with raw ticket fields (subject, description,
            error_logs, stack_trace, product_module)

    Returns:
        DataFrame with integer-encoded features (int_* columns) plus
        original category, subcategory, and _count columns
    """
    result = df[['category', 'subcategory']].copy()

    if '_count' in df.columns:
        result['_count'] = df['_count']
    else:
        result['_count'] = 1

    # Apply text transformations
    df['_error_logs_nots'] = df['error_logs'].apply(remove_timestamps_from_logs)
    df['_err_code'] = df['description'].apply(lambda x: process_description(x)[0])
    df['_description'] = df['description'].apply(lambda x: process_description(x)[1])
    df['_subject'] = df.apply(lambda row: mask_subject(row['subject'], row['_err_code']), axis=1)

    # Integer-encode all features
    features_to_encode = [
        ('_err_code', 'int__err_code'),
        ('_subject', 'int__subject'),
        ('_description', 'int__description'),
        ('product_module', 'int_product_module'),
        ('_error_logs_nots', 'int__error_logs_nots'),
        ('stack_trace', 'int_stack_trace'),
    ]

    for source_col, int_col in features_to_encode:
        if source_col in df.columns:
            codes, _ = encode_column_to_int(df[source_col])
            result[int_col] = codes

    return result


if __name__ == '__main__':
    # Demonstrate usage
    print("Example: Encoding categorical features for statistical tests")
    print("\nThis module provides utilities to prepare ticket data for")
    print("chi-squared tests, mutual information analysis, and ML classifiers")
    print("that require integer-encoded categorical features.")
    print("\nSee investigate_subcategory.py for full usage.")
