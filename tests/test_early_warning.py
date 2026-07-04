"""
tests/test_early_warning.py
Automated tests for the Week 6 early warning model.

Run with:
    python tests/test_early_warning.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))

from api.early_warning import predict_stress_risk, evaluate_on_dataset

MODEL_PATH  = Path("models/isolation_forest_live.pkl")
SCALER_PATH = Path("models/scaler_live.pkl")


def test_model_files_exist():
    assert MODEL_PATH.exists(), \
        f"{MODEL_PATH} missing — run api/anomaly_detection_v2.py first"
    assert SCALER_PATH.exists(), \
        f"{SCALER_PATH} missing — run api/anomaly_detection_v2.py first"
    print("Test 1 passed: model files exist")


def test_prediction_returns_correct_keys():
    reading = {"pH": 6.0, "EC": 1.5, "water_temp_C": 20.0}
    result  = predict_stress_risk(reading)
    required = {"stress_risk", "risk_probability", "risk_level",
                "anomaly_score", "message"}
    assert required.issubset(set(result.keys())), \
        f"Missing keys: {required - set(result.keys())}"
    print("Test 2 passed: prediction returns all required keys")


def test_probability_is_between_0_and_1():
    reading = {"pH": 6.0, "EC": 1.5, "water_temp_C": 20.0}
    result  = predict_stress_risk(reading)
    assert 0.0 <= result["risk_probability"] <= 1.0, \
        f"Probability out of range: {result['risk_probability']}"
    print(f"Test 3 passed: probability is valid ({result['risk_probability']})")


def test_risk_level_is_valid():
    reading = {"pH": 6.0, "EC": 1.5, "water_temp_C": 20.0}
    result  = predict_stress_risk(reading)
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}, \
        f"Unknown risk level: {result['risk_level']}"
    print(f"Test 4 passed: risk level is valid ({result['risk_level']})")


def test_bad_readings_get_higher_risk_than_normal():
    """
    Anomalous readings (pH crash, EC spike, temp spike) must get a
    higher average risk probability than normal readings.
    Uses averages over 3 readings of each type for stability.
    """
    normal_readings = [
        {"pH": 6.00, "EC": 1.50, "water_temp_C": 20.0},
        {"pH": 5.95, "EC": 1.48, "water_temp_C": 19.5},
        {"pH": 6.05, "EC": 1.52, "water_temp_C": 20.5},
    ]
    bad_readings = [
        {"pH": 3.9,  "EC": 3.50, "water_temp_C": 32.0},
        {"pH": 8.5,  "EC": 0.40, "water_temp_C": 33.0},
        {"pH": 4.1,  "EC": 3.40, "water_temp_C": 31.5},
    ]

    avg_normal = sum(predict_stress_risk(r)["risk_probability"]
                     for r in normal_readings) / len(normal_readings)
    avg_bad    = sum(predict_stress_risk(r)["risk_probability"]
                     for r in bad_readings)    / len(bad_readings)

    assert avg_bad > avg_normal, (
        f"Bad readings avg risk ({avg_bad:.4f}) must be > "
        f"normal readings avg risk ({avg_normal:.4f})"
    )
    print(f"Test 5 passed: bad readings avg risk ({avg_bad:.4f}) > "
          f"normal readings avg risk ({avg_normal:.4f})")


def test_roc_auc_above_baseline():
    """
    On the full labeled dataset, anomaly rows must get higher risk scores
    than normal rows — measured by ROC-AUC (>0.5 = better than random).
    """
    results = evaluate_on_dataset()
    assert results["roc_auc"] > 0.5, \
        f"ROC-AUC {results['roc_auc']} is not above baseline 0.5"
    assert results["anomaly_avg_risk"] > results["normal_avg_risk"], \
        (f"Anomaly avg risk ({results['anomaly_avg_risk']}) must be > "
         f"normal avg risk ({results['normal_avg_risk']})")
    print(f"Test 6 passed: ROC-AUC={results['roc_auc']}, "
          f"anomaly avg risk={results['anomaly_avg_risk']} > "
          f"normal avg risk={results['normal_avg_risk']}")


if __name__ == "__main__":
    test_model_files_exist()
    test_prediction_returns_correct_keys()
    test_probability_is_between_0_and_1()
    test_risk_level_is_valid()
    test_bad_readings_get_higher_risk_than_normal()
    test_roc_auc_above_baseline()
    print("\nAll Week 6 early warning tests passed!")