"""Create clean, balanced train/val/test split from dbt-processed tickets.

This module extracts unique subject→category mappings (~110 samples), balances
them to equal representation per category, and splits into stratified train/val/test.

All models train on this clean data (not the noisy 100K full corpus).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_from_duckdb(duckdb_path: Path, table: str = "featured_tickets") -> pd.DataFrame:
    """Load processed tickets from dbt output table."""
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {table}").fetchdf()
        return df
    finally:
        con.close()


def extract_clean_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique subject→category mappings.

    The key insight: subject→category mapping is CLEAN (no conflicting labels).
    Each unique subject maps to exactly one category across all 100K tickets.

    Args:
        df: Full tickets DataFrame with 'subject' and 'category' columns

    Returns:
        DataFrame with one row per unique subject (~110 samples)
    """
    # Verify no conflicting labels
    subject_groups = df.groupby("subject")["category"].nunique()
    conflicts = subject_groups[subject_groups > 1]

    if len(conflicts) > 0:
        print(f"WARNING: Found {len(conflicts)} subjects with conflicting categories")
        raise ValueError(
            "Subject→Category mapping has conflicts! Clean data assumption violated."
        )

    # Extract first occurrence of each subject
    sample_indices = df.groupby("subject")["ticket_id"].idxmin()
    clean_df = df.loc[sample_indices].copy()

    print(f"Extracted {len(clean_df)} unique subjects from {len(df)} tickets")

    # Show category distribution
    cat_counts = clean_df["category"].value_counts()
    print("\nCategory distribution (before balancing):")
    for cat in sorted(cat_counts.index):
        print(f"  {cat}: {cat_counts[cat]} subjects")

    return clean_df


def balance_categories(
    df: pd.DataFrame,
    target_per_category: int = 22,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance dataset so each category has equal representation.

    Uses oversampling (with replacement) for minority classes and undersampling
    (without replacement) for majority classes.

    Args:
        df: DataFrame with 'category' column
        target_per_category: Number of samples per category (default 22, median)
        random_state: Random seed for reproducibility

    Returns:
        Balanced DataFrame
    """
    rng = np.random.default_rng(random_state)
    balanced_dfs = []

    print(f"\nBalancing to {target_per_category} samples per category...")

    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat].copy()
        n_current = len(cat_df)

        if n_current < target_per_category:
            # Oversample with replacement
            sampled = cat_df.sample(n=target_per_category, replace=True, random_state=random_state)
            print(f"  {cat}: {n_current} → {target_per_category} (oversampled)")
        elif n_current > target_per_category:
            # Undersample without replacement
            sampled = cat_df.sample(n=target_per_category, replace=False, random_state=random_state)
            print(f"  {cat}: {n_current} → {target_per_category} (undersampled)")
        else:
            sampled = cat_df
            print(f"  {cat}: {n_current} (no change)")

        balanced_dfs.append(sampled)

    result = pd.concat(balanced_dfs, ignore_index=True)

    # Shuffle to mix categories
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return result


def stratified_split(
    df: pd.DataFrame,
    stratify_column: str = "category",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = config.SPLIT_SEED,
) -> pd.DataFrame:
    """
    Split DataFrame into train/val/test using stratified sampling.

    Ensures each split has proportional representation of all categories.

    Args:
        df: Source DataFrame with stratify_column
        stratify_column: Column to stratify by (default "category")
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for test (default 0.15)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with 'split' column added ('train', 'val', or 'test')
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0: "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    print(f"\nSplitting with stratification by '{stratify_column}'")

    # First split: train vs (val + test)
    temp_size = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_state,
        stratify=df[stratify_column],
    )

    # Second split: val vs test
    test_size_adjusted = test_ratio / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size_adjusted,
        random_state=random_state,
        stratify=temp_df[stratify_column],
    )

    # Add split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    result = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Show split statistics
    print(f"\nSplit sizes:")
    print(f"  train: {len(train_df)} ({100*len(train_df)/len(result):.1f}%)")
    print(f"  val:   {len(val_df)} ({100*len(val_df)/len(result):.1f}%)")
    print(f"  test:  {len(test_df)} ({100*len(test_df)/len(result):.1f}%)")

    print(f"\nCategory distribution by split:")
    for split_name in ["train", "val", "test"]:
        split_df = result[result["split"] == split_name]
        counts = split_df[stratify_column].value_counts()
        print(f"  {split_name}: {dict(sorted(counts.items()))}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create clean, balanced, stratified train/val/test split"
    )
    parser.add_argument(
        "--duckdb-path",
        type=str,
        default=str(config.DUCKDB_PATH),
        help="Path to DuckDB file with dbt output",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="featured_tickets",
        help="dbt output table name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.CLEAN_TRAINING_PARQUET_PATH),
        help="Output parquet path for clean training data",
    )
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=22,
        help="Target samples per category after balancing",
    )
    args = parser.parse_args()

    config.ensure_directories()

    # Load full corpus from dbt
    print(f"Loading from DuckDB: {args.duckdb_path}")
    df_full = load_from_duckdb(Path(args.duckdb_path), args.table)
    print(f"Loaded {len(df_full)} total tickets")

    # Extract clean subject→category mappings
    df_clean = extract_clean_subjects(df_full)

    # Balance categories
    df_balanced = balance_categories(df_clean, target_per_category=args.target_per_category)
    print(f"\nBalanced dataset: {len(df_balanced)} samples")

    # Stratified split
    df_split = stratified_split(df_balanced)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_split.to_parquet(output_path, index=False)
    print(f"\nSaved clean training data: {output_path}")
    print(f"Total samples: {len(df_split)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
