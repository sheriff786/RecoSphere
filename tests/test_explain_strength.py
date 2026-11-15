"""P0.5 item 2: users see a reason's strength (its share of the item's positive SHAP), never raw weights."""
import pytest

from serve.product import MEDIUM, STRONG, strength


@pytest.mark.parametrize("share,label", [(1.0, "strong"), (STRONG, "strong"), (0.399, "medium"), (MEDIUM, "medium"),
                                         (0.199, "weak"), (0.0, "weak")])
def test_strength_thresholds(share, label):
    assert strength(share) == label


def test_thresholds_are_ordered():
    assert 0 < MEDIUM < STRONG <= 1
