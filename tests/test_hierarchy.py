from src.models import hierarchy


def test_derive_category():
    assert hierarchy.derive_category("Bug") == "Technical Issue"
    assert hierarchy.derive_category("Billing") == "Account Management"
    assert hierarchy.derive_category("Encryption") == "Security"


def test_get_all_categories():
    categories = hierarchy.get_all_categories()
    assert len(categories) == 5
    assert "Technical Issue" in categories
    assert "Security" in categories


def test_get_all_priorities():
    priorities = hierarchy.get_all_priorities()
    assert priorities == ["low", "medium", "high", "critical"]


def test_get_all_sentiments():
    sentiments = hierarchy.get_all_sentiments()
    assert len(sentiments) == 6
    assert "neutral" in sentiments
    assert "frustrated" in sentiments
