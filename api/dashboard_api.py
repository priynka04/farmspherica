from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
from datetime import datetime

app = FastAPI(title="Farmspherica Dashboard API")

# This allows your frontend (Streamlit) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "data/farmspherica.db"

# Safe ranges for each sensor — used to trigger alerts
SAFE_RANGES = {
    "pH":           (4.0, 9.0),
    "EC":           (0.0, 5.0),
    "water_temp_C": (10,  35),
}

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

# ── Endpoint 3: alerts ──────────────────────────────────────────────────────
@app.get("/alerts")
def get_alerts():
    """Checks the latest reading and returns any out-of-range alerts."""
    conn = get_db()
    df = pd.read_sql("SELECT * FROM sensor_readings ORDER BY rowid DESC LIMIT 1", conn)
    conn.close()
    if df.empty:
        return {"alerts": []}
    row = df.iloc[0]
    alerts = []
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