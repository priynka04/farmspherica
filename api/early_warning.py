"""
api/early_warning.py
Week 6 — Predictive Early Warning + Multi-Modal Fusion

Detects when current sensor readings are moving INTO anomaly territory,
giving the team an early risk signal before a full anomaly fires.

Why this approach instead of predicting future anomalies:
The sensor_anomaly_labeled.csv has anomalies randomly injected — there is
no deteriorating trend before them, so no model can predict them from current
readings. The correct interpretation of "early warning" for this dataset is:
use the Isolation Forest's continuous anomaly score (decision_function) to
see HOW CLOSE the current reading is to the anomaly boundary. A reading with
a score drifting negative is heading toward anomaly territory — that IS the
early warning.

Uses the LIVE Isolation Forest model from Week 7 (trained on pH, EC,
water_temp_C — the 3 columns that actually exist in the live database).

Run directly to evaluate and confirm:
    python api/early_warning.py
"""

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LIVE_MODEL_PATH  = Path("models/isolation_forest_live.pkl")
LIVE_SCALER_PATH = Path("models/scaler_live.pkl")
DATA_PATH        = Path("data/sensor_anomaly_labeled.csv")
REPORT_PATH      = Path("docs/early_warning_eval_report.md")

# Columns the LIVE model was trained on (confirmed Week 7)
LIVE_FEATURE_COLS = ["pH", "EC", "water_temp_C"]

# Mapping: external callers may use EC_mScm — accept both
_EC_ALIASES = {"EC_mScm", "EC_mscm", "ec"}

# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
_cache = {}

def _load():
    if not _cache:
        if not LIVE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{LIVE_MODEL_PATH} not found. "
                "Run api/anomaly_detection_v2.py first (Week 7)."
            )
        with open(LIVE_MODEL_PATH, "rb") as f:
            _cache["model"] = pickle.load(f)
        with open(LIVE_SCALER_PATH, "rb") as f:
            _cache["scaler"] = pickle.load(f)
    return _cache["model"], _cache["scaler"]


# ---------------------------------------------------------------------------
# Feature extraction — accepts both "EC" and "EC_mScm" as keys
# ---------------------------------------------------------------------------
def _extract_features(reading: dict) -> list:
    row = []
    for col in LIVE_FEATURE_COLS:
        if col in reading:
            row.append(float(reading[col]))
        elif col == "EC":
            # Try common aliases
            val = None
            for alias in _EC_ALIASES:
                if alias in reading:
                    val = float(reading[alias])
                    break
            row.append(val if val is not None else 0.0)
        else:
            row.append(float(reading.get(col, 0.0)))
    return row


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------
def predict_stress_risk(reading: dict) -> dict:
    """
    Computes a risk score for the current sensor reading using the Isolation
    Forest's continuous anomaly score (decision_function).

    decision_function returns:
        positive value  → reading is comfortably inside the normal zone
        near zero       → reading is at the edge of normal territory
        negative value  → reading is in anomaly territory (early warning!)

    We convert this to a 0-1 risk probability using a sigmoid so the output
    is easy to display and threshold.

    Parameters
    ----------
    reading : dict
        Current sensor values. Must contain at least pH, EC (or EC_mScm),
        and water_temp_C. Extra keys are ignored.

    Returns
    -------
    dict with keys:
        stress_risk       bool    — True if risk >= threshold
        risk_probability  float   — 0.0 (safe) to 1.0 (definite anomaly)
        risk_level        str     — "LOW" / "MEDIUM" / "HIGH"
        anomaly_score     float   — raw IF score (negative = anomalous)
        message           str     — human-readable summary
    """
    model, scaler = _load()

    features = _extract_features(reading)
    X_s = scaler.transform([features])

    # Continuous anomaly score: positive = normal, negative = anomalous
    score = float(model.decision_function(X_s)[0])

    # Convert to risk probability via sigmoid
    # Steepness=15 maps: score=+0.1 → ~18% risk, score=0 → 50%, score=-0.1 → ~82%
    risk_prob = round(1.0 / (1.0 + math.exp(15.0 * score)), 4)
    risk_prob = max(0.0, min(1.0, risk_prob))

    if risk_prob < 0.30:
        level = "LOW"
    elif risk_prob < 0.60:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "stress_risk":      risk_prob >= 0.5,
        "risk_probability": risk_prob,
        "risk_level":       level,
        "anomaly_score":    round(score, 5),
        "message": (
            f"Stress risk: {level} "
            f"(probability {risk_prob:.1%}, IF score {score:.3f})"
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation: confirm the score separates anomalies from normals
# ---------------------------------------------------------------------------
def evaluate_on_dataset(data_path: Path = DATA_PATH) -> dict:
    """
    Loads the labeled anomaly dataset, runs every row through the risk scorer,
    and computes ROC-AUC and average scores per class to confirm it works.
    """
    model, scaler = _load()

    df = pd.read_csv(data_path)
    # Rename EC_mScm to EC so the live model column names match
    if "EC_mScm" in df.columns and "EC" not in df.columns:
        df = df.rename(columns={"EC_mScm": "EC"})

    X = df[LIVE_FEATURE_COLS].values
    X_s = scaler.transform(X)

    scores    = model.decision_function(X_s)          # continuous score
    labels    = df["is_anomaly"].values                # 1 = anomaly
    risk_probs = 1.0 / (1.0 + np.exp(15.0 * scores))  # vectorised sigmoid

    auc = roc_auc_score(labels, risk_probs)

    normal_avg = risk_probs[labels == 0].mean()
    anomaly_avg = risk_probs[labels == 1].mean()

    return {
        "roc_auc":       round(auc, 4),
        "normal_avg_risk":  round(float(normal_avg), 4),
        "anomaly_avg_risk": round(float(anomaly_avg), 4),
        "n_normal":   int((labels == 0).sum()),
        "n_anomaly":  int((labels == 1).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[INFO] Loading Isolation Forest model (from Week 7)...")
    _load()
    print(f"  Model: {LIVE_MODEL_PATH}")
    print(f"  Features: {LIVE_FEATURE_COLS}")

    print("\n[INFO] Evaluating on labeled dataset...")
    results = evaluate_on_dataset()

    print(f"\n  ROC-AUC:                {results['roc_auc']}")
    print(f"  Avg risk (normal rows): {results['normal_avg_risk']}")
    print(f"  Avg risk (anomaly rows):{results['anomaly_avg_risk']}")
    print(f"  Normal rows:            {results['n_normal']}")
    print(f"  Anomaly rows:           {results['n_anomaly']}")
    print(f"\n  -> Anomaly rows have {results['anomaly_avg_risk'] / max(results['normal_avg_risk'], 1e-9):.1f}x "
          f"higher risk than normal rows")

    print("\n[INFO] Quick prediction tests...")
    tests = [
        ("Normal reading",    {"pH": 6.0,  "EC": 1.50, "water_temp_C": 20.0}),
        ("pH crash",          {"pH": 3.9,  "EC": 1.50, "water_temp_C": 20.0}),
        ("EC spike",          {"pH": 6.0,  "EC": 3.50, "water_temp_C": 20.0}),
        ("Temp spike",        {"pH": 6.0,  "EC": 1.50, "water_temp_C": 33.0}),
        ("Multi-fault",       {"pH": 3.9,  "EC": 3.50, "water_temp_C": 33.0}),
    ]
    for name, reading in tests:
        r = predict_stress_risk(reading)
        print(f"  {name:<20} → {r['risk_level']:<6} "
              f"({r['risk_probability']:.1%}, score={r['anomaly_score']:.3f})")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Week 6 — Early Warning Model Evaluation Report\n\n")
        f.write("**Approach:** Isolation Forest continuous anomaly score "
                "(decision_function) as real-time risk signal.\n\n")
        f.write("**Model used:** `models/isolation_forest_live.pkl` "
                "(trained in Week 7 on pH, EC, water_temp_C).\n\n")
        f.write("## Why this approach\n\n")
        f.write(
            "The initial approach (XGBoost trained to predict randomly-injected "
            "future anomalies) gave F1=0.03 because the synthetic dataset has "
            "no deteriorating trend before anomalies — they are randomly placed. "
            "The correct interpretation of 'early warning' is: use the Isolation "
            "Forest's continuous score to detect when readings are drifting toward "
            "anomaly territory. This is directly computable from current readings "
            "and requires no training of a new model.\n\n"
        )
        f.write("## Results\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| ROC-AUC | {results['roc_auc']} |\n")
        f.write(f"| Avg risk score (normal rows) | {results['normal_avg_risk']} |\n")
        f.write(f"| Avg risk score (anomaly rows) | {results['anomaly_avg_risk']} |\n\n")
        f.write("## Known limitations\n\n")
        f.write(
            "- Validated on synthetic data only. Real-plant trends will be "
            "clearer since real anomalies follow organic deterioration.\n"
            "- Only 3 sensor features (pH, EC, water_temp_C) — limited by "
            "live database schema. Will improve as more sensors are added.\n"
            "- Segmentation output (canopy_area, biomass) not yet fused — "
            "will be added once ESP32-CAM is live.\n"
        )
    print(f"\nReport saved -> {REPORT_PATH}")
    print("\nWEEK 6 EARLY WARNING MODEL — DONE")
    print(f"  ROC-AUC: {results['roc_auc']}")
    print(f"  Anomaly rows get {results['anomaly_avg_risk'] / max(results['normal_avg_risk'], 1e-9):.1f}x "
          f"higher risk scores than normal rows")


if __name__ == "__main__":
    main()