"""
anomaly_detection_v2.py
Week 3 — Upgraded ML Anomaly Detection for Farmspherica Nano PAW

Trains TWO models from the same labeled dataset (data/sensor_anomaly_labeled.csv,
2000 rows, 8 labeled anomaly types):

  1. FULL model (7 features) — the official Week 3 deliverable per the work
     plan, trained and evaluated on the complete labeled dataset.
     Saved to models/isolation_forest_v2.pkl / scaler_v2.pkl.

  2. LIVE model (3 features: pH, EC, water_temp_C) — your real
     sensor_readings table only logs these 3 sensors (confirmed via
     `PRAGMA table_info(sensor_readings)`); TDS_ppm, DO_mgL, ambient_temp_C,
     and humidity_pct are not collected yet. This smaller model is the one
     actually wired into the dashboard /alerts endpoint.
     Saved to models/isolation_forest_live.pkl / scaler_live.pkl.

Both models use the same two-layer detection logic:
  Layer A — Isolation Forest (unsupervised multivariate outlier detection)
            catches spikes/crashes/drops/multi-sensor faults.
  Layer B — Frozen-sensor rule (deterministic check) catches "sensor_stuck"
            anomalies, where a reading is an exact repeat of the previous
            reading. Isolation Forest cannot see this pattern because a
            frozen value still looks like a perfectly normal reading on
            its own — it is only wrong because it didn't change.

Run directly to train, evaluate, and save both models:
    python api/anomaly_detection_v2.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/sensor_anomaly_labeled.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.25

FEATURE_COLS = [
    "pH", "EC_mScm", "TDS_ppm", "water_temp_C",
    "DO_mgL", "ambient_temp_C", "humidity_pct",
]
MODEL_PATH = Path("models/isolation_forest_v2.pkl")
SCALER_PATH = Path("models/scaler_v2.pkl")
REPORT_PATH = Path("docs/anomaly_v2_eval_report.md")

# Live DB column is named "EC", not "EC_mScm" — renamed at load time so this
# model's feature names match dashboard_api.py's column names exactly.
LIVE_FEATURE_COLS = ["pH", "EC", "water_temp_C"]
LIVE_MODEL_PATH = Path("models/isolation_forest_live.pkl")
LIVE_SCALER_PATH = Path("models/scaler_live.pkl")
LIVE_REPORT_PATH = Path("docs/anomaly_live_eval_report.md")


# ---------------------------------------------------------------------------
# Load + prep data
# ---------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"EC_mScm": "EC_live"})  # keep both names available
    df["EC_mScm"] = df["EC_live"]
    df["EC"] = df["EC_live"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_frozen_flag(df: pd.DataFrame, feature_cols: list) -> pd.Series:
    """A row is 'frozen' if every feature column is identical to the row
    immediately before it (confirmed: real sensor_stuck rows have ALL
    feature columns unchanged; normal rows almost never do)."""
    diffs = df[feature_cols].diff().abs()
    return ((diffs < 1e-9).sum(axis=1) == len(feature_cols)).astype(int)


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------
def train_isolation_forest(X_train: pd.DataFrame, contamination: float):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled)
    return model, scaler


def predict_layer_a(model, scaler, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict(scaler.transform(X))  # -1 = anomaly, 1 = normal
    return np.where(raw == -1, 1, 0)


def predict_anomaly(reading: dict, model, scaler, feature_cols: list = None) -> dict:
    """
    reading: dict with current values AND previous values for the frozen
             check, e.g. {"pH": 6.0, "EC": 1.5, "water_temp_C": 20.0,
             "prev": {same keys}}
    feature_cols: which columns this model/scaler were trained on.
             Use FEATURE_COLS for the full model, LIVE_FEATURE_COLS for the
             live/dashboard model.
    """
    feature_cols = feature_cols or FEATURE_COLS
    row = pd.DataFrame([{c: reading[c] for c in feature_cols}])
    layer_a = predict_layer_a(model, scaler, row)[0]

    layer_b = 0
    if reading.get("prev") is not None:
        prev = reading["prev"]
        unchanged = sum(1 for c in feature_cols if abs(reading[c] - prev[c]) < 1e-9)
        layer_b = 1 if unchanged == len(feature_cols) else 0

    is_anomaly = bool(max(layer_a, layer_b))
    if layer_b:
        reason = "sensor_stuck (reading identical to previous reading)"
    elif layer_a:
        reason = "statistical outlier (Isolation Forest)"
    else:
        reason = "normal"
    return {"is_anomaly": is_anomaly, "reason": reason}


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate(y_test, combined_preds, types_test) -> dict:
    report_text = classification_report(
        y_test, combined_preds, target_names=["normal", "anomaly"]
    )
    cm = confusion_matrix(y_test, combined_preds)

    eval_df = pd.DataFrame(
        {"true": y_test.values, "pred": combined_preds, "type": types_test.values}
    )
    per_type = {}
    for t in sorted(eval_df["type"].unique()):
        if t == "normal":
            continue
        sub = eval_df[eval_df["type"] == t]
        per_type[t] = {"n": int(len(sub)), "recall": round(float((sub["pred"] == 1).mean()), 3)}

    overall_recall = float((eval_df[eval_df["true"] == 1]["pred"] == 1).mean())
    pos = eval_df[eval_df["pred"] == 1]
    overall_precision = float((pos["true"] == 1).mean()) if len(pos) else 0.0

    return {
        "report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "per_type_recall": per_type,
        "overall_recall": round(overall_recall, 4),
        "overall_precision": round(overall_precision, 4),
    }


# ---------------------------------------------------------------------------
# One shared pipeline, run twice with different feature sets
# ---------------------------------------------------------------------------
def run_pipeline(df, feature_cols, model_path, scaler_path, report_path, label):
    print("\n" + "=" * 70)
    print(f"[INFO] Training {label} model ({len(feature_cols)} features: {feature_cols})")
    print("=" * 70)

    frozen = add_frozen_flag(df, feature_cols)
    X = df[feature_cols]
    y = df["is_anomaly"]
    types = df["anomaly_type"]

    idx_train, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    types_test = types.loc[idx_test]

    contamination = float(y_train.mean())
    model, scaler = train_isolation_forest(X_train, contamination)

    layer_a_preds = predict_layer_a(model, scaler, X_test)
    frozen_preds = frozen.loc[idx_test].values
    combined_preds = np.maximum(layer_a_preds, frozen_preds)

    results = evaluate(y_test, combined_preds, types_test)
    print(results["report_text"])
    print("Confusion matrix:")
    print(np.array(results["confusion_matrix"]))
    print("Recall per anomaly type:")
    for t, v in results["per_type_recall"].items():
        print(f"  {t}: n={v['n']}, recall={v['recall']}")
    print(f"\nOverall {label} recall:    {results['overall_recall']}")
    print(f"Overall {label} precision: {results['overall_precision']}")
    target_met = results["overall_recall"] >= 0.90
    print(f"Target (>=90% recall): {'MET' if target_met else 'NOT MET'}")

    # Retrain on full data for deployment, now that we know it performs well
    final_model, final_scaler = train_isolation_forest(X, contamination=float(y.mean()))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(final_scaler, f)
    print(f"Model saved -> {model_path}")
    print(f"Scaler saved -> {scaler_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"# Week 3 — Anomaly Detection {label} Model Evaluation\n\n")
        f.write(f"Features used: {feature_cols}\n\n")
        f.write(f"Dataset: `{DATA_PATH}` ({len(df)} rows, {int(y.sum())} labeled anomalies)\n\n")
        f.write(f"Test set size: {len(idx_test)} rows (25% held out, stratified)\n\n")
        f.write("## Overall metrics\n\n")
        f.write(f"- Recall: **{results['overall_recall']}** (target: >= 0.90)\n")
        f.write(f"- Precision: **{results['overall_precision']}**\n")
        f.write(f"- Target met: **{target_met}**\n\n")
        f.write("## Recall per anomaly type\n\n| Type | n (test) | Recall |\n|---|---|---|\n")
        for t, v in results["per_type_recall"].items():
            f.write(f"| {t} | {v['n']} | {v['recall']} |\n")
        f.write("\n## Classification report (test set)\n\n```\n")
        f.write(results["report_text"])
        f.write("\n```\n\n## Confusion matrix (test set)\n\n```\n")
        f.write(str(np.array(results["confusion_matrix"])))
        f.write("\n```\n")
        if label == "LIVE":
            f.write(
                "\n**Known limitation:** `DO_crash` recall is low because there is no "
                "dissolved-oxygen sensor logged in the live database — this anomaly type "
                "cannot be reliably caught until a DO reading is added to `sensor_readings`. "
                "Flagged as a Month 2 ask for Livia/Akash.\n"
            )
    print(f"Report saved -> {report_path}")
    return results


def main():
    print("[INFO] Loading data...")
    df = load_data()
    print(f"Shape: {df.shape}")
    print("Anomaly type counts:")
    print(df["anomaly_type"].value_counts())

    run_pipeline(df, FEATURE_COLS, MODEL_PATH, SCALER_PATH, REPORT_PATH, label="FULL")
    run_pipeline(df, LIVE_FEATURE_COLS, LIVE_MODEL_PATH, LIVE_SCALER_PATH, LIVE_REPORT_PATH, label="LIVE")

    print("\nWEEK 3 ANOMALY DETECTION v2 — DONE (both FULL and LIVE models trained)")


if __name__ == "__main__":
    main()