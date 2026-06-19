"""
tests/test_anomaly_v2.py
Automated tests for the Week 3 anomaly models (both FULL and LIVE versions).

Run with:
    python tests/test_anomaly_v2.py
"""

import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))

from api.anomaly_detection_v2 import predict_anomaly, FEATURE_COLS, LIVE_FEATURE_COLS

MODEL_PATH = Path("models/isolation_forest_v2.pkl")
SCALER_PATH = Path("models/scaler_v2.pkl")
LIVE_MODEL_PATH = Path("models/isolation_forest_live.pkl")
LIVE_SCALER_PATH = Path("models/scaler_live.pkl")


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_model_files_exist():
    for p in [MODEL_PATH, SCALER_PATH, LIVE_MODEL_PATH, LIVE_SCALER_PATH]:
        assert p.exists(), f"{p} is missing — run api/anomaly_detection_v2.py first"
    print("Test 1 passed: all model files exist (FULL + LIVE)")


def test_normal_reading_not_flagged(model, scaler):
    reading = {
        "pH": 6.00, "EC_mScm": 1.50, "TDS_ppm": 1000, "water_temp_C": 20.0,
        "DO_mgL": 6.8, "ambient_temp_C": 23.0, "humidity_pct": 60.0,
        "prev": {
            "pH": 5.95, "EC_mScm": 1.48, "TDS_ppm": 990, "water_temp_C": 19.8,
            "DO_mgL": 6.7, "ambient_temp_C": 22.9, "humidity_pct": 59.0,
        },
    }
    result = predict_anomaly(reading, model, scaler, FEATURE_COLS)
    assert result["is_anomaly"] is False, f"Expected normal, got {result}"
    print("Test 2 passed: normal reading correctly NOT flagged (FULL model)")


def test_ph_crash_flagged(model, scaler):
    reading = {
        "pH": 3.9, "EC_mScm": 1.50, "TDS_ppm": 1000, "water_temp_C": 20.0,
        "DO_mgL": 6.8, "ambient_temp_C": 23.0, "humidity_pct": 60.0,
        "prev": {
            "pH": 6.0, "EC_mScm": 1.50, "TDS_ppm": 1000, "water_temp_C": 20.0,
            "DO_mgL": 6.8, "ambient_temp_C": 23.0, "humidity_pct": 60.0,
        },
    }
    result = predict_anomaly(reading, model, scaler, FEATURE_COLS)
    assert result["is_anomaly"] is True, f"Expected anomaly, got {result}"
    print("Test 3 passed: pH crash correctly flagged (FULL model)")


def test_sensor_stuck_flagged(model, scaler):
    repeated = {"pH": 6.00, "EC_mScm": 1.50, "TDS_ppm": 1000, "water_temp_C": 20.0,
                "DO_mgL": 6.8, "ambient_temp_C": 23.0, "humidity_pct": 60.0}
    reading = dict(repeated)
    reading["prev"] = dict(repeated)
    result = predict_anomaly(reading, model, scaler, FEATURE_COLS)
    assert result["is_anomaly"] is True
    assert "sensor_stuck" in result["reason"]
    print("Test 4 passed: frozen/stuck sensor correctly flagged (FULL model)")


def test_live_model_ph_crash_flagged(live_model, live_scaler):
    reading = {
        "pH": 3.9, "EC": 1.50, "water_temp_C": 20.0,
        "prev": {"pH": 6.0, "EC": 1.50, "water_temp_C": 20.0},
    }
    result = predict_anomaly(reading, live_model, live_scaler, LIVE_FEATURE_COLS)
    assert result["is_anomaly"] is True, f"Expected anomaly, got {result}"
    print("Test 5 passed: pH crash correctly flagged (LIVE 3-feature model)")


if __name__ == "__main__":
    test_model_files_exist()
    model, scaler = load(MODEL_PATH), load(SCALER_PATH)
    live_model, live_scaler = load(LIVE_MODEL_PATH), load(LIVE_SCALER_PATH)

    test_normal_reading_not_flagged(model, scaler)
    test_ph_crash_flagged(model, scaler)
    test_sensor_stuck_flagged(model, scaler)
    test_live_model_ph_crash_flagged(live_model, live_scaler)

    print("\nAll anomaly detection v2 tests passed!")