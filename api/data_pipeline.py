import pandas as pd
import sqlite3
from datetime import datetime

# ---- STEP 1: READ THE CSV ----
def read_csv(filepath):
    """Read the CSV file into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df

# ---- STEP 2: VALIDATE DATA ----
def validate_data(df):
    """Check for missing values and out-of-range sensor readings."""
    issues = []

    # Check for missing values
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            issues.append(f"Missing values in '{col}': {count} rows")

    # Check value ranges (adjust column names to match Livia's schema)
    range_checks = {
    'pH':               (4.0,  9.0),
    'TDS':              (0,    5000),
    'EC':               (0.0,  5.0),
    'water_temp_C':     (10,   35),
    'water_level_cm':   (0,    100),
    'air_temp_C':       (10,   45),
    'humidity_pct':     (0,    100),
    'light_hours':      (0,    24),
    'plant_height_cm':  (0,    300),
    'leaf_count':       (0,    500),
}

    for col, (min_val, max_val) in range_checks.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                issues.append(f"Out-of-range values in '{col}': {len(out_of_range)} rows")
                # Flag them with a new column
                df[f'{col}_flag'] = (df[col] < min_val) | (df[col] > max_val)

    # Print all issues found
    if issues:
        print("VALIDATION ISSUES FOUND:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✅ Data validation passed — no issues found.")

    return df, issues

# ---- STEP 3: SAVE TO SQLITE DATABASE ----
def save_to_db(df, db_path="data/farmspherica.db", table_name="sensor_readings"):
    """Save the validated DataFrame to a SQLite database."""
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()
    print(f"✅ Saved {len(df)} rows to database: {db_path}")

# ---- MAIN FUNCTION ----
def run_pipeline(csv_filepath):
    df = read_csv(csv_filepath)
    df_validated, issues = validate_data(df)
    save_to_db(df_validated)
    return df_validated, issues

# Run it
if __name__ == "__main__":
    run_pipeline("data/sample_data.csv")