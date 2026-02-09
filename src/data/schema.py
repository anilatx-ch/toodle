"""Pandera schema validation for processed dataset."""

from __future__ import annotations

import argparse

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src import config


def build_schema() -> DataFrameSchema:
    return DataFrameSchema(
        {
            "ticket_id": Column(str, unique=True, nullable=False),
            "category": Column(str, nullable=False),
            "priority_ordinal": Column(int, checks=Check.isin([0, 1, 2, 3])),
            "severity_ordinal": Column(int, checks=Check.isin([0, 1, 2, 3, 4])),
            "account_age_days": Column(int, checks=Check.in_range(30, 1000), nullable=False),
            "account_monthly_value": Column(float, checks=Check.ge(0), nullable=False),
            "split": Column(str, checks=Check.isin(["train", "val", "test"]), nullable=False),
            "text_combined_tfidf": Column(str, checks=Check.str_length(1), nullable=False),
            "text_combined_bert": Column(str, checks=Check.str_length(1), nullable=False),
            "has_leakage_pattern": Column(bool, nullable=False),
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
        default=str(config.SPLIT_PARQUET_PATH),
        help="Input parquet path",
    )
    args = parser.parse_args()

    validated = validate_parquet(args.input)
    leakage_count = int(validated["has_leakage_pattern"].sum())

    print(f"Validation passed for {len(validated)} rows")
    print(
        "Leakage warning: "
        f"{leakage_count} rows flagged "
        f"({100.0 * leakage_count / max(len(validated), 1):.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
