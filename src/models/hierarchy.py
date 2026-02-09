"""Category/subcategory hierarchy utilities and supported target labels."""

from __future__ import annotations

from typing import Dict, List, Tuple

CATEGORY_TO_SUBCATEGORIES: Dict[str, List[str]] = {
    "Account Management": [
        "Access Control",
        "Billing",
        "License",
        "Subscription",
        "Upgrade",
    ],
    "Data Issue": ["Corruption", "Data Loss", "Import/Export", "Sync Error", "Validation"],
    "Feature Request": ["API", "Documentation", "Enhancement", "New Feature", "UI/UX"],
    "Security": ["Authentication", "Authorization", "Compliance", "Encryption", "Vulnerability"],
    "Technical Issue": ["Bug", "Compatibility", "Configuration", "Integration", "Performance"],
}

SUBCATEGORY_TO_CATEGORY: Dict[str, str] = {
    subcategory: category
    for category, subcategories in CATEGORY_TO_SUBCATEGORIES.items()
    for subcategory in subcategories
}

PRIORITY_LEVELS: Tuple[str, ...] = ("low", "medium", "high", "critical")
SENTIMENT_LEVELS: Tuple[str, ...] = (
    "angry",
    "confused",
    "frustrated",
    "grateful",
    "neutral",
    "satisfied",
)


def derive_category(subcategory: str) -> str:
    """Return parent category for a subcategory."""
    if subcategory not in SUBCATEGORY_TO_CATEGORY:
        raise KeyError(f"Unknown subcategory: {subcategory}")
    return SUBCATEGORY_TO_CATEGORY[subcategory]


def get_all_categories() -> List[str]:
    """Return a sorted list of all supported categories."""
    return sorted(CATEGORY_TO_SUBCATEGORIES.keys())


def get_all_priorities() -> List[str]:
    """Return priority labels used by the priority classifier."""
    return list(PRIORITY_LEVELS)


def get_all_sentiments() -> List[str]:
    """Return sentiment labels used by the sentiment classifier."""
    return list(SENTIMENT_LEVELS)
