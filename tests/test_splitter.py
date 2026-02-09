import pandas as pd
import pytest

from src.data.splitter import balance_categories, extract_clean_subjects, stratified_split


def test_extract_clean_subjects():
    # Create test data with unique subject→category mappings
    df = pd.DataFrame({
        "ticket_id": ["t1", "t2", "t3", "t4", "t5"],
        "subject": ["Bug in login", "Bug in login", "Feature X", "Feature X", "Security"],
        "category": ["Technical Issue", "Technical Issue", "Feature Request", "Feature Request", "Security"],
        "description": ["desc1", "desc2", "desc3", "desc4", "desc5"],
    })

    result = extract_clean_subjects(df)

    assert len(result) == 3  # 3 unique subjects
    assert set(result["subject"]) == {"Bug in login", "Feature X", "Security"}


def test_extract_clean_subjects_conflict():
    # Create test data with conflicting labels
    df = pd.DataFrame({
        "ticket_id": ["t1", "t2"],
        "subject": ["Bug in login", "Bug in login"],
        "category": ["Technical Issue", "Feature Request"],  # Conflict!
    })

    with pytest.raises(ValueError, match="conflicts"):
        extract_clean_subjects(df)


def test_balance_categories():
    df = pd.DataFrame({
        "category": ["A"] * 10 + ["B"] * 2 + ["C"] * 5,
        "value": range(17),
    })

    balanced = balance_categories(df, target_per_category=5, random_state=42)

    assert len(balanced) == 15  # 5 per category * 3 categories
    assert (balanced["category"].value_counts() == 5).all()


def test_stratified_split():
    df = pd.DataFrame({
        "category": ["A"] * 100 + ["B"] * 100,
        "value": range(200),
    })

    result = stratified_split(df, stratify_column="category", random_state=42)

    assert "split" in result.columns
    assert set(result["split"]) == {"train", "val", "test"}

    # Check proportions
    train_count = (result["split"] == "train").sum()
    val_count = (result["split"] == "val").sum()
    test_count = (result["split"] == "test").sum()

    assert 130 <= train_count <= 150  # ~70%
    assert 25 <= val_count <= 35      # ~15%
    assert 25 <= test_count <= 35     # ~15%

    # Check stratification: each split should have both categories
    for split_name in ["train", "val", "test"]:
        split_df = result[result["split"] == split_name]
        assert len(split_df["category"].unique()) == 2
