from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
import pickle  # NEW — needed to load the saved Week 3 model
import sys
from pathlib import Path
from datetime import datetime

# NEW — makes "from anomaly_detection_v2 import ..." work no matter how this
# server is started (uvicorn from project root, or from inside api/)
sys.path.append(str(Path(__file__).resolve().parent))
from anomaly_detection_v2 import predict_anomaly, LIVE_FEATURE_COLS as ML_FEATURE_COLS  # NEW

app = FastAPI(title="Farmspherica Dashboard API")

# This allows your frontend (Streamlit) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "data/farmspherica.db"

# Safe ranges for each sensor — used to trigger alerts (Layer 1, unchanged)
SAFE_RANGES = {
    "pH":           (4.0, 9.0),
    "EC":           (0.0, 5.0),
    "water_temp_C": (10,  35),
}

# ── NEW — Week 3 ML anomaly model (Layer 2), loaded once at startup ────────
# Using the LIVE model (3 features), not the 7-feature one — your real
# sensor_readings table only has pH, EC, water_temp_C (checked via
# PRAGMA table_info). ML_FEATURE_COLS imported above is already
# ["pH", "EC", "water_temp_C"], matching these DB column names exactly, so
# no renaming is needed when building the reading dict below.
# See docs/anomaly_live_eval_report.md for this model's accuracy.
ANOMALY_MODEL_PATH = Path("models/isolation_forest_live.pkl")
ANOMALY_SCALER_PATH = Path("models/scaler_live.pkl")

try:
    with open(ANOMALY_MODEL_PATH, "rb") as f:
        anomaly_model_v2 = pickle.load(f)
    with open(ANOMALY_SCALER_PATH, "rb") as f:
        anomaly_scaler_v2 = pickle.load(f)
    print("[INFO] Week 3 ML anomaly model (live, 3-feature) loaded successfully.")
except FileNotFoundError:
    anomaly_model_v2 = None
    anomaly_scaler_v2 = None
    print("[WARNING] models/isolation_forest_live.pkl not found — "
          "run api/anomaly_detection_v2.py first. /alerts will fall back "
          "to rule-based checks only.")


def get_db():
    """Opens a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn

# ── Endpoint 1: latest sensor reading ──────────────────────────────────────
@app.get("/data/latest")
def get_latest():
    """Returns the most recent row of sensor data."""
    conn = get_db()
    df = pd.read_sql("SELECT * FROM sensor_readings ORDER BY rowid DESC LIMIT 1", conn)
    conn.close()
    if df.empty:
        return {"error": "No data found"}
    return df.iloc[0].to_dict()

# ── Endpoint 2: 7-day trend data ────────────────────────────────────────────
@app.get("/data/trends")
def get_trends():
    """Returns the last 7 rows for trend charts."""
    conn = get_db()
    df = pd.read_sql("SELECT * FROM sensor_readings ORDER BY rowid DESC LIMIT 7", conn)
    conn.close()
    df = df.iloc[::-1]  # reverse so oldest is first (left side of chart)
    return df.to_dict(orient="records")

# ── Endpoint 3: alerts (Layer 1 rule-based + Layer 2 ML — UPDATED) ──────────
@app.get("/alerts")
def get_alerts():
    """Checks the latest reading and returns any out-of-range or ML-flagged alerts."""
    conn = get_db()
    # CHANGED: was LIMIT 1 — now LIMIT 2 so we have the previous reading too,
    # which the ML model needs for the frozen/stuck-sensor check.
    df = pd.read_sql("SELECT * FROM sensor_readings ORDER BY rowid DESC LIMIT 2", conn)
    conn.close()
    if df.empty:
        return {"alerts": []}

    row = df.iloc[0]                                # latest reading
    prev_row = df.iloc[1] if len(df) > 1 else None    # reading right before it

    alerts = []

    # Layer 1 — rule-based safe range check (unchanged from before)
    for col, (low, high) in SAFE_RANGES.items():
        if col in row and pd.notna(row[col]):
            val = float(row[col])
            if val < low:
                alerts.append({
                    "sensor": col,
                    "value": val,
                    "message": f"{col} is {val} — below safe minimum of {low}"
                })
            elif val > high:
                alerts.append({
                    "sensor": col,
                    "value": val,
                    "message": f"{col} is {val} — above safe maximum of {high}"
                })

    # Layer 2 — NEW — Week 3 ML anomaly model (live, 3-feature version)
    if anomaly_model_v2 is not None and all(
        col in row.keys() and pd.notna(row[col]) for col in ML_FEATURE_COLS
    ):
        reading = {col: float(row[col]) for col in ML_FEATURE_COLS}

        if prev_row is not None and all(
            col in prev_row.keys() and pd.notna(prev_row[col]) for col in ML_FEATURE_COLS
        ):
            reading["prev"] = {col: float(prev_row[col]) for col in ML_FEATURE_COLS}
        else:
            reading["prev"] = None

        ml_result = predict_anomaly(
            reading, anomaly_model_v2, anomaly_scaler_v2, feature_cols=ML_FEATURE_COLS
        )
        if ml_result["is_anomaly"]:
            alerts.append({
                "sensor": "ML_MODEL",
                "value": None,
                "message": f"ML anomaly model flagged this reading — {ml_result['reason']}"
            })

    return {"alerts": alerts, "count": len(alerts)}

# ── Endpoint 4: all plant records ───────────────────────────────────────────
@app.get("/plants")
def get_plants():
    """Returns all rows — for the plant growth table."""
    conn = get_db()
    df = pd.read_sql("SELECT * FROM sensor_readings ORDER BY rowid ASC", conn)
    conn.close()
    records = []
    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            try:
                if pd.isna(val):
                    record[col] = None
                else:
                    record[col] = val
            except (TypeError, ValueError):
                record[col] = val
        records.append(record)
    return {"total": len(records), "records": records}

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}