import pandas as pd
import os

# ── Read the real strawberry data from the uploaded Excel file ──────────────
# Change this path to wherever you saved the file on your computer
EXCEL_PATH = "data/strawberry records.xlsx"

# First copy the uploaded file into your data/ folder:
# Just drag strawberry_records.xlsx into your data/ folder in VS Code

df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")

print("Original columns found:")
print(df.columns.tolist())
print(f"\nRows: {len(df)}")
print("\nRaw data preview:")
print(df.head(7))

# ── Rename columns to match the official NanoPAW schema ────────────────────
df = df.rename(columns={
    'Date':                 'date',
    'Day':                  'day_number',
    'Plant height (cm)':    'plant_height_cm',
    'Leaf count ':          'leaf_count',
    'Root colour ':         'root_colour',
    'Water colour':         'water_colour',
    'odor/smell':           'water_smell',
    'EC/Ppm (ms/cm)':       'EC',
    'Ph':                   'pH',
    'Water temperature ˚c': 'water_temp_C',
    'notes':                'remarks',
    'magnesium sulphate':   'nutrient_MgSO4',
    'clacium nitrate':      'nutrient_CaNO3',
})

# ── Add missing columns with sensible defaults ──────────────────────────────
df['plant_id']        = 'P01'
df['week_number']     = 1
df['observer']        = 'Ambika'
df['condition']       = 'Healthy'
df['deficiency_type'] = 'None'
df['stress_symptoms'] = 'None'
df['deficiency_type'] = df['deficiency_type'].fillna('None')
df['stress_symptoms'] = df['stress_symptoms'].fillna('None')
df['nutrient_MgSO4']  = df['nutrient_MgSO4'].fillna('0g')
df['nutrient_CaNO3']  = df['nutrient_CaNO3'].fillna('0g')
df['nutrient_formula']= 'Standard_v1'

# ── Fill the date column (was empty in the file) ────────────────────────────
df['date'] = pd.date_range(
    start='2026-05-20', periods=len(df)
).strftime('%Y-%m-%d')

# ── Save as a clean CSV in the data/ folder ─────────────────────────────────
os.makedirs('data', exist_ok=True)
output_path = 'data/strawberry_real_data.csv'
df.to_csv(output_path, index=False)

print(f"\nSaved {len(df)} rows to: {output_path}")
print("\nFinal columns:")
print(df.columns.tolist())
print("\nClean data preview:")
print(df[['date', 'day_number', 'plant_id', 'pH', 'EC',
          'water_temp_C', 'plant_height_cm', 'leaf_count',
          'condition', 'remarks']].to_string())