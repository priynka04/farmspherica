"""
tests/test_shap_explainer.py
Tests for the Week 8 SHAP explainability module.

Run with:
    python tests/test_shap_explainer.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))

from api.shap_explainer import explain_prediction, global_feature_importance


def test_explain_returns_correct_keys():
    reading = {
        "day_after_transplant": 30, "pH": 6.0,
        "EC_mScm": 1.5, "leaf_count": 10,
        "water_temp_C": 20.0,
    }
    result = explain_prediction(reading)
    required = {"predicted_height_cm", "top_positive", "top_negative",
                 "bottleneck", "explanation", "shap_values"}
    assert required.issubset(set(result.keys())), \
        f"Missing keys: {required - set(result.keys())}"
    print("Test 1 passed: explain_prediction returns all required keys")


def test_predicted_height_is_reasonable():
    reading = {
        "day_after_transplant": 30, "pH": 6.0,
        "EC_mScm": 1.5, "leaf_count": 10,
        "water_temp_C": 20.0,
    }
    result = explain_prediction(reading)
    assert 1.0 <= result["predicted_height_cm"] <= 100.0, \
        f"Height {result['predicted_height_cm']} is out of reasonable range"
    print(f"Test 2 passed: predicted height is reasonable "
          f"({result['predicted_height_cm']} cm)")


def test_low_ec_gives_lower_height_than_normal():
    normal = explain_prediction({
        "day_after_transplant": 30, "pH": 6.0,
        "EC_mScm": 1.5, "leaf_count": 10, "water_temp_C": 20.0,
    })
    low_ec = explain_prediction({
        "day_after_transplant": 30, "pH": 6.0,
        "EC_mScm": 0.3, "leaf_count": 4, "water_temp_C": 20.0,
    })
    assert low_ec["predicted_height_cm"] <= normal["predicted_height_cm"], (
        f"Low EC plant ({low_ec['predicted_height_cm']} cm) should be "
        f"<= normal plant ({normal['predicted_height_cm']} cm)"
    )
    print(f"Test 3 passed: low EC gives lower/equal height "
          f"({low_ec['predicted_height_cm']} vs {normal['predicted_height_cm']} cm)")


def test_shap_values_sum_is_nonzero():
    reading = {
        "day_after_transplant": 30, "pH": 6.0,
        "EC_mScm": 1.5, "leaf_count": 10, "water_temp_C": 20.0,
    }
    result = explain_prediction(reading)
    total  = sum(abs(v) for v in result["shap_values"].values())
    assert total > 0, "All SHAP values are zero — something went wrong"
    print(f"Test 4 passed: SHAP values are non-zero (total |SHAP|={total:.3f})")


def test_global_importance_has_all_features():
    importance = global_feature_importance(n_samples=50)
    assert len(importance) == 13, \
        f"Expected 13 features, got {len(importance)}"
    assert "feature" in importance.columns
    assert "mean_abs_shap" in importance.columns
    top = importance.iloc[0]["feature"]
    print(f"Test 5 passed: global importance has 13 features "
          f"(top: {top})")


if __name__ == "__main__":
    test_explain_returns_correct_keys()
    test_predicted_height_is_reasonable()
    test_low_ec_gives_lower_height_than_normal()
    test_shap_values_sum_is_nonzero()
    test_global_importance_has_all_features()
    print("\nAll Week 8 SHAP explainability tests passed!")