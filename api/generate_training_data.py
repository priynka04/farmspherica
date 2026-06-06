import pandas as pd
import numpy as np
import sqlite3

DB_PATH = "data/farmspherica.db"

np.random.seed(42)
n = 50

data = {
    "date":            pd.date_range("2026-05-20", periods=n).strftime("%Y-%m-%d"),
    "day_number":      range(1, n + 1),
    "plant_id":        ["P01"] * n,
    "pH":              np.clip(np.random.normal(6.0, 0.25, n), 5.0, 7.5),
    "EC":              np.clip(np.random.normal(1.2, 0.2,  n), 0.5, 2.5),
    "water_temp_C":    np.clip(np.random.normal(21.0, 1.0, n), 17,  26),
    "plant_height_cm": np.linspace(12.5, 22.0, n) + np.random.normal(0, 0.3, n),
    "leaf_count":      np.clip(np.round(np.linspace(4, 14, n)), 4, 20).astype(int),
    "condition":       ["Healthy"] * n,
    "week_number":     [(i // 7) + 1 for i in range(n)],
    "observer":        ["Ambika"] * n,
    "deficiency_type": ["None"] * n,
    "stress_symptoms": ["None"] * n,
    "nutrient_formula":["Standard_v1"] * n,
}

# Inject 5 anomalies so Isolation Forest learns what bad data looks like
anomaly_indices = [10, 20, 30, 38, 45]
for i in anomaly_indices:
    data["pH"][i]        = float(np.random.choice([3.5, 8.5]))
    data["EC"][i]        = float(np.random.choice([0.2, 4.5]))
    data["condition"][i] = "Mildly Stressed"

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data/extended_training_data.csv", index=False)
print(f"Saved {len(df)} rows to data/extended_training_data.csv")

# Add to database
conn = sqlite3.connect(DB_PATH)
df.to_sql("sensor_readings", conn, if_exists="append", index=False)

total = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
conn.close()

print(f"Added to database. Total rows now in sensor_readings: {total}")
print("Done!")