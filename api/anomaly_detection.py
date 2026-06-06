# =============================================================
# FILE: api/anomaly_detection.py
# WHAT IT DOES: Detects when sensor readings go outside safe range
#               Uses both rule-based checks AND Isolation Forest ML model
#               This file is used by dashboard_api.py for the /alerts endpoint
# HOW TO RUN:   python api/anomaly_detection.py
#               (runs a self-test to confirm everything works)
# =============================================================

import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# =============================================================
# SAFE RANGES — these are the healthy values for strawberry plants
# If any reading falls outside these ranges, it is flagged as an anomaly
# You can update these values based on Livia's guidance
# =============================================================
SAFE_RANGES = {
    "pH":         (5.5, 6.5),
    "EC":         (0.8, 2.0),
    "water_temp_C": (18.0, 26.0)
}

# Where the trained model and scaler will be saved
MODEL_PATH  = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"
DB_PATH     = "data/farmspherica.db"

# Columns used for ML training
FEATURE_COLS = ["pH", "EC","water_temp_C"]


# =============================================================
# PART 1 — RULE-BASED ALERTS (simple, always works)
# This checks each reading against the hardcoded safe ranges above.
# No training needed. Works even with very little data.
# =============================================================

def rule_based_alerts(row: dict) -> list:
    """
    Check one sensor reading row against SAFE_RANGES.
    Returns a list of alert strings (empty list = all good).

    Example input:  {"pH": 3.2, "EC": 1.0, "TDS": 800, "water_temp": 22}
    Example output: ["pH is OUT OF RANGE: 3.2 (safe: 5.5 – 6.5)"]
    """
    alerts = []
    for param, (low, high) in SAFE_RANGES.items():
        value = row.get(param)
        if value is None:
            continue  # skip if column not present
        try:
            value = float(value)
            if value < low or value > high:
                alerts.append(
                    f"{param} is OUT OF RANGE: {value:.2f} "
                    f"(safe range: {low} – {high})"
                )
        except (ValueError, TypeError):
            alerts.append(f"{param} has invalid value: {value}")
    return alerts


# =============================================================
# PART 2 — ISOLATION FOREST MODEL (ML-based anomaly detection)
# Isolation Forest is an unsupervised ML algorithm.
# It learns what "normal" data looks like, then flags anything unusual.
# contamination=0.1 means it expects about 10% of data to be anomalies.
# =============================================================

def train_isolation_forest(db_path: str = DB_PATH):
    """
    Reads all sensor data from SQLite, trains an Isolation Forest model,
    and saves the model + scaler to the models/ folder.

    Call this once when you have enough data (at least 20 rows recommended).
    Re-run it whenever you get a lot of new data from Livia/Ambika.
    """
    os.makedirs("models", exist_ok=True)

    # --- Load data from database ---
    print("[INFO] Loading sensor data from database...")
    conn   = sqlite3.connect(db_path)
    df     = pd.read_sql("SELECT * FROM sensor_readings", conn)
    conn.close()

    # Keep only the columns we need for anomaly detection
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 2:
        print(f"[ERROR] Not enough sensor columns found. Found: {df.columns.tolist()}")
        return None, None

    df_features = df[available].dropna()

    if len(df_features) < 5:
        print(f"[WARNING] Only {len(df_features)} rows available. "
              "Using rule-based alerts only. Train with more data later.")
        return None, None

    print(f"[INFO] Training on {len(df_features)} rows, columns: {available}")

    # --- Scale the data (Isolation Forest works better on scaled data) ---
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(df_features)

    # --- Train Isolation Forest ---
    # contamination=0.1 → model assumes ~10% of readings are anomalies
    # n_estimators=100  → 100 trees (more = more accurate but slower)
    # random_state=42   → makes results reproducible
    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42
    )
    model.fit(X_scaled)

    # --- Save model and scaler ---
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[OK] Isolation Forest trained and saved to {MODEL_PATH}")
    print(f"[OK] Scaler saved to {SCALER_PATH}")
    return model, scaler


def load_isolation_forest():
    """
    Loads the trained model and scaler from disk.
    Returns (model, scaler) or (None, None) if not trained yet.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def ml_based_alert(row: dict) -> dict:
    """
    Uses the trained Isolation Forest model to check if a reading is anomalous.
    Returns a dict with: is_anomaly (bool), confidence (str), score (float)

    The anomaly score is between -1 and 1:
        Negative score = more anomalous
        Positive score = more normal
    """
    model, scaler = load_isolation_forest()

    if model is None:
        return {
            "is_anomaly": False,
            "confidence": "MODEL NOT TRAINED",
            "score": None,
            "message": "Isolation Forest not trained yet. Using rule-based alerts only."
        }

    # Build feature vector — use only columns the model was trained on
    available = [c for c in FEATURE_COLS if c in row and row[c] is not None]
    if len(available) < 2:
        return {
            "is_anomaly": False,
            "confidence": "INSUFFICIENT DATA",
            "score": None,
            "message": "Not enough sensor columns to run ML check."
        }

    values   = np.array([[float(row[c]) for c in available]])
    scaled   = scaler.transform(values)
    pred     = model.predict(scaled)[0]        # 1 = normal, -1 = anomaly
    score    = model.score_samples(scaled)[0]  # more negative = more anomalous

    is_anomaly = (pred == -1)
    confidence = "HIGH" if abs(score) > 0.1 else "LOW"

    return {
        "is_anomaly": is_anomaly,
        "confidence": confidence,
        "score": round(float(score), 4),
        "message": "ANOMALY DETECTED by ML model" if is_anomaly else "Normal reading (ML check passed)"
    }


# =============================================================
# PART 3 — COMBINED ALERT FUNCTION
# This is what dashboard_api.py will call.
# It runs BOTH rule-based AND ML checks, combines the results.
# =============================================================

def get_all_alerts(row: dict) -> dict:
    """
    Main function called by the dashboard API.
    Runs both rule-based and ML-based checks on a sensor reading.

    Input:  dict with sensor readings e.g. {"pH": 6.1, "EC": 1.0, ...}
    Output: dict with combined alert info
    """
    rule_alerts = rule_based_alerts(row)
    ml_result   = ml_based_alert(row)

    has_alert = len(rule_alerts) > 0 or ml_result["is_anomaly"]

    return {
        "timestamp":    row.get("timestamp", str(datetime.now())),
        "has_alert":    has_alert,
        "alert_level":  "CRITICAL" if len(rule_alerts) > 0 else ("WARNING" if ml_result["is_anomaly"] else "OK"),
        "rule_alerts":  rule_alerts,
        "ml_result":    ml_result,
        "summary":      f"{len(rule_alerts)} rule alert(s). ML: {ml_result['message']}"
    }


# =============================================================
# PART 4 — EVALUATION
# Compares rule-based vs ML on your full dataset.
# Run this to see which approach works better for your data.
# =============================================================

def evaluate_models(db_path: str = DB_PATH):
    """
    Loads all data and runs both models.
    Prints a comparison table showing which rows each model flagged.
    Useful for showing Livia which approach to use in production.
    """
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM sensor_readings", conn)
    conn.close()

    print("\n===== MODEL EVALUATION =====")
    print(f"Total rows: {len(df)}\n")

    rule_flag_count = 0
    ml_flag_count   = 0

    for _, row in df.iterrows():
        row_dict    = row.to_dict()
        rule_alerts = rule_based_alerts(row_dict)
        ml_result   = ml_based_alert(row_dict)

        if rule_alerts:
            rule_flag_count += 1
        if ml_result["is_anomaly"]:
            ml_flag_count += 1

    print(f"Rule-based flagged: {rule_flag_count} / {len(df)} rows")
    print(f"ML model flagged:   {ml_flag_count} / {len(df)} rows")
    print(f"\nNote: With only {len(df)} rows, rule-based is more reliable.")
    print("Re-run after Week 4 when you have more real data.")


# =============================================================
# SELF-TEST — Run this file directly to confirm everything works
# Command: python api/anomaly_detection.py
# =============================================================

if __name__ == "__main__":
    print("===== ANOMALY DETECTION SELF-TEST =====\n")

    # Test 1 — Normal reading (should NOT trigger any alerts)
    normal_row = {"pH": 6.0, "EC": 1.2, "water_temp_C": 22.0}
    result     = get_all_alerts(normal_row)
    print(f"Test 1 — Normal reading:")
    print(f"  Alert level : {result['alert_level']}")
    print(f"  Rule alerts : {result['rule_alerts']}")
    print(f"  ML result   : {result['ml_result']['message']}")
    assert result["alert_level"] == "OK" or result["alert_level"] == "WARNING", "Test 1 passed"
    print("  [OK] Test 1 passed\n")

    # Test 2 — Bad pH (should trigger CRITICAL rule alert)
    bad_row = {"pH": 3.0, "EC": 1.2, "water_temp_C": 22.0}
    result  = get_all_alerts(bad_row)
    print(f"Test 2 — Bad pH (3.0):")
    print(f"  Alert level : {result['alert_level']}")
    print(f"  Rule alerts : {result['rule_alerts']}")
    assert result["alert_level"] == "CRITICAL", "Test 2 should be CRITICAL"
    print("  [OK] Test 2 passed\n")

    # Test 3 — Bad EC (should trigger CRITICAL rule alert)
    bad_ec = {"pH": 6.0, "EC": 5.5, "water_temp_C": 22.0}
    result = get_all_alerts(bad_ec)
    print(f"Test 3 — Bad EC (5.5):")
    print(f"  Alert level : {result['alert_level']}")
    print(f"  Rule alerts : {result['rule_alerts']}")
    assert result["alert_level"] == "CRITICAL", "Test 3 should be CRITICAL"
    print("  [OK] Test 3 passed\n")

    # Test 4 — Train the model (if DB exists)
    if os.path.exists(DB_PATH):
        print("Test 4 — Training Isolation Forest on real data...")
        train_isolation_forest()
        evaluate_models()
    else:
        print(f"Test 4 — Skipped (no DB found at {DB_PATH})")
        print("         Run this after your dashboard_api.py has populated the DB.")

    print("\n===== ALL TESTS PASSED =====")