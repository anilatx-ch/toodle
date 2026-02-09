"""Pandera schema validation for processed dataset."""

from __future__ import annotations

import argparse

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src import config


def build_schema() -> DataFrameSchema:
    """Build validation schema for clean training data (before feature engineering).

    Note: ticket_id is not unique because balancing may oversample rows.
    """
    return DataFrameSchema(
        {
            "ticket_id": Column(str, nullable=False),
            "category": Column(str, nullable=False),
            "subcategory": Column(str, nullable=False),
            "subject": Column(str, checks=Check.str_length(1), nullable=False),
            "description": Column(str, checks=Check.str_length(1), nullable=False),
            "priority": Column(str, nullable=False),
            "severity": Column(str, nullable=False),
            "account_age_days": Column(int, checks=Check.in_range(30, 1000), nullable=False),
            "account_monthly_value": Column(float, checks=Check.ge(0), nullable=False),
            "split": Column(str, checks=Check.isin(["train", "val", "test"]), nullable=False),
        },
        coerce=True,
        strict=False,
    )


def validate_parquet(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    schema = build_schema()
    return schema.validate(df)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate featured parquet using Pandera")
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.CLEAN_TRAINING_PARQUET_PATH),
        help="Input parquet path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.SPLIT_PARQUET_PATH),
        help="Output parquet path for validated data",
    )
    args = parser.parse_args()

    validated = validate_parquet(args.input)

    print(f"Validation passed for {len(validated)} rows")

    # Show split distribution
    split_counts = validated["split"].value_counts()
    print(f"Split distribution: {dict(sorted(split_counts.items()))}")

    # Write validated data to output
    validated.to_parquet(args.output, index=False)
    print(f"Saved validated data to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
