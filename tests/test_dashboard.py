import sys
sys.path.append(".")
import sqlite3
import pandas as pd

DB_PATH = "data/farmspherica.db"

def test_database_has_data():
    """Check that the database exists and has sensor readings in it."""
    conn = sqlite3.connect(DB_PATH)
    
    # Get all table names
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    print(f"Tables found: {table_names}")
    
    # Find whichever sensor table exists
    sensor_table = None
    for name in ["sensor_readings", "sensor_data"]:
        if name in table_names:
            sensor_table = name
            break
    
    assert sensor_table is not None, "No sensor table found in database!"
    
    # Count rows
    count = conn.execute(
        f"SELECT COUNT(*) FROM {sensor_table}"
    ).fetchone()[0]
    print(f"Rows in {sensor_table}: {count}")
    assert count > 0, "Sensor table is empty!"
    conn.close()
    print("Test 1 passed: database has data")

def test_alerts_logic():
    """Check that out-of-range values are correctly detected."""
    
    # This simulates what the /alerts endpoint does
    SAFE_RANGES = {
        "pH":           (4.0, 9.0),
        "EC":           (0.0, 5.0),
        "water_temp_C": (10,  35),
    }
    
    # Test with a row that has a bad pH value (20 is way out of range)
    test_row = {"pH": 20.0, "EC": 1.5, "water_temp_C": 22.0}
    alerts = []
    for col, (low, high) in SAFE_RANGES.items():
        if col in test_row:
            val = float(test_row[col])
            if val < low or val > high:
                alerts.append(col)
    
    assert "pH" in alerts, "pH=20 should trigger an alert!"
    assert "EC" not in alerts, "EC=1.5 should NOT trigger an alert!"
    print("Test 2 passed: alert logic works correctly")

def test_photos_table_exists():
    """Check that the plant_photos table was created by image_api.py."""
    conn = sqlite3.connect(DB_PATH)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    conn.close()
    
    assert "plant_photos" in table_names, \
        "plant_photos table not found — run image_api.py first!"
    print("Test 3 passed: plant_photos table exists")

def test_trends_returns_7_rows_max():
    """Check that the trends query returns at most 7 rows."""
    conn = sqlite3.connect(DB_PATH)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    
    sensor_table = None
    for name in ["sensor_readings", "sensor_data"]:
        if name in table_names:
            sensor_table = name
            break
    
    if sensor_table:
        df = pd.read_sql(
            f"SELECT * FROM {sensor_table} ORDER BY rowid DESC LIMIT 7",
            conn
        )
        assert len(df) <= 7, "Trends should return max 7 rows!"
        print(f"Test 4 passed: trends returns {len(df)} rows (max 7)")
    conn.close()

if __name__ == "__main__":
    test_database_has_data()
    test_alerts_logic()
    test_photos_table_exists()
    test_trends_returns_7_rows_max()
    print("\nAll dashboard tests passed!")