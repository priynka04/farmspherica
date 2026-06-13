# =============================================================
# FILE: api/growth_model.py
# WHAT IT DOES: Trains Random Forest + XGBoost to predict
#               plant_height_cm and fresh_weight_g from sensor data.
#               Saves the best model to models/growth_model.pkl
# HOW TO RUN:   python api/growth_model.py
# =============================================================

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

os.makedirs("models", exist_ok=True)
os.makedirs("docs",   exist_ok=True)

# =============================================================
# STEP 1 — Load both datasets
# =============================================================

print("[INFO] Loading datasets...")
lettuce    = pd.read_csv("data/lollo_rosa_lettuce_synthetic.csv")
strawberry = pd.read_csv("data/strawberry_synthetic.csv")

# Add a crop_type column so the model knows which crop it is
lettuce["crop_type"]    = "lettuce"
strawberry["crop_type"] = "strawberry"

# Combine both into one big dataset
df = pd.concat([lettuce, strawberry], ignore_index=True)
print(f"[INFO] Combined dataset: {len(df)} rows")

# =============================================================
# STEP 2 — Prepare features
# These are the sensor readings the model will learn from
# =============================================================

# Encode growth_stage as a number (establishment=0, vegetative=1, etc.)
le = LabelEncoder()
df["growth_stage_encoded"] = le.fit_transform(df["growth_stage"].fillna("unknown"))

# Encode crop_type as a number (lettuce=0, strawberry=1)
df["crop_type_encoded"] = (df["crop_type"] == "strawberry").astype(int)

# Features the model uses as input
FEATURES = [
    "day_after_transplant",
    "water_temp_C",
    "ambient_temp_C",
    "pH",
    "EC_mScm",
    "TDS_ppm",
    "DO_mgL",
    "humidity_pct",
    "photoperiod_hr",
    "PPFD_umol",
    "leaf_count",
    "growth_stage_encoded",
    "crop_type_encoded",
]

# What we want to predict
TARGET_HEIGHT = "plant_height_cm"

# Keep only rows where all features and target exist
df_clean = df[FEATURES + [TARGET_HEIGHT]].dropna()
X = df_clean[FEATURES]
y = df_clean[TARGET_HEIGHT]

print(f"[INFO] Training rows: {len(X)}")
print(f"[INFO] Features: {FEATURES}")

# =============================================================
# STEP 3 — Split into train and test (80% train, 20% test)
# =============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[INFO] Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# =============================================================
# STEP 4 — Train Random Forest
# Random Forest builds 200 decision trees and averages their predictions
# =============================================================

print("\n[INFO] Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2   = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"  Random Forest — R²: {rf_r2:.4f} | RMSE: {rf_rmse:.4f} cm")

# =============================================================
# STEP 5 — Train XGBoost
# XGBoost builds trees sequentially, each one fixing errors from the last
# =============================================================

print("\n[INFO] Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_r2   = r2_score(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
print(f"  XGBoost       — R²: {xgb_r2:.4f} | RMSE: {xgb_rmse:.4f} cm")

# =============================================================
# STEP 6 — Compare both models and save the better one
# R² closer to 1.0 = better. RMSE lower = better.
# =============================================================

print("\n===== MODEL COMPARISON =====")
print(f"  Random Forest — R²: {rf_r2:.4f} | RMSE: {rf_rmse:.2f} cm")
print(f"  XGBoost       — R²: {xgb_r2:.4f} | RMSE: {xgb_rmse:.2f} cm")

if rf_r2 >= xgb_r2:
    best_model = rf
    best_name  = "Random Forest"
    best_r2    = rf_r2
    best_rmse  = rf_rmse
    best_pred  = rf_pred
else:
    best_model = xgb_model
    best_name  = "XGBoost"
    best_r2    = xgb_r2
    best_rmse  = xgb_rmse
    best_pred  = xgb_pred

print(f"\n  Winner: {best_name} (R²={best_r2:.4f})")

# Save model, feature list, and label encoder
joblib.dump(best_model, "models/growth_model.pkl")
joblib.dump(FEATURES,   "models/growth_model_features.pkl")
joblib.dump(le,         "models/growth_stage_encoder.pkl")
print(f"  Model saved → models/growth_model.pkl")

# =============================================================
# STEP 7 — Feature importance chart
# Shows which sensor matters most for predicting height
# =============================================================

if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "feature":    FEATURES,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"], color="#2e7d32")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig("docs/growth_feature_importance.png", dpi=150)
    plt.close()
    print("  Feature importance chart saved → docs/growth_feature_importance.png")

# =============================================================
# STEP 8 — Actual vs Predicted chart
# =============================================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_pred, alpha=0.5, color="#1565c0", s=20)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect prediction")
plt.xlabel("Actual height (cm)")
plt.ylabel("Predicted height (cm)")
plt.title(f"Actual vs Predicted — {best_name} | R²={best_r2:.4f}")
plt.legend()
plt.tight_layout()
plt.savefig("docs/growth_actual_vs_predicted.png", dpi=150)
plt.close()
print("  Actual vs Predicted chart saved → docs/growth_actual_vs_predicted.png")

# =============================================================
# STEP 9 — Quick prediction function (used by dashboard)
# =============================================================

def predict_growth(day, water_temp, ambient_temp, ph, ec, tds,
                   do, humidity, photoperiod, ppfd, leaf_count,
                   growth_stage="vegetative", crop_type="lettuce"):
    """
    Predicts plant height given sensor readings.
    Call this from the dashboard to show the forecast panel.
    """
    model    = joblib.load("models/growth_model.pkl")
    features = joblib.load("models/growth_model_features.pkl")
    encoder  = joblib.load("models/growth_stage_encoder.pkl")

    try:
        stage_encoded = encoder.transform([growth_stage])[0]
    except Exception:
        stage_encoded = 1

    crop_encoded = 1 if crop_type == "strawberry" else 0

    row = [[
        day, water_temp, ambient_temp, ph, ec, tds,
        do, humidity, photoperiod, ppfd, leaf_count,
        stage_encoded, crop_encoded
    ]]

    prediction = model.predict(row)[0]
    return round(float(prediction), 2)


# =============================================================
# FINAL SUMMARY
# =============================================================

print(f"""
{'='*50}
  WEEK 1 COMPLETE — Growth Prediction Model
  Best model  : {best_name}
  R² score    : {best_r2:.4f}  (target: ≥ 0.80)
  RMSE        : {best_rmse:.2f} cm
  Saved to    : models/growth_model.pkl
  Charts      : docs/growth_feature_importance.png
                docs/growth_actual_vs_predicted.png

  Write in your report:
  Growth Prediction Model ({best_name})
  Trained on {len(X_train)} rows | R²={best_r2:.4f} | RMSE={best_rmse:.2f}cm
{'='*50}
""")

if __name__ == "__main__":
    pass