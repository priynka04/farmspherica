import sys
sys.path.append(".")
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def test_model_files_exist():
    assert os.path.exists("models/growth_model.pkl"), "Model not found!"
    assert os.path.exists("models/growth_model_features.pkl"), "Features not found!"
    print("Test 1 passed: model files exist")

def test_model_predicts_reasonable_height():
    model = joblib.load("models/growth_model.pkl")
    # day=10, water_temp=20, ambient_temp=21, pH=6.0, EC=1.2, TDS=800,
    # DO=7.0, humidity=65, photoperiod=14, PPFD=220, leaf_count=8,
    # growth_stage_encoded=1, crop_type_encoded=0 (lettuce)
    sample = [[10, 20.0, 21.0, 6.0, 1.2, 800, 7.0, 65.0, 14.0, 220, 8, 1, 0]]
    pred   = model.predict(sample)[0]
    assert 2 <= pred <= 30, f"Prediction out of reasonable range: {pred}"
    print(f"Test 2 passed: predicted height = {pred:.1f} cm (reasonable)")

def test_model_r2_above_threshold():
    lettuce    = pd.read_csv("data/lollo_rosa_lettuce_synthetic.csv")
    strawberry = pd.read_csv("data/strawberry_synthetic.csv")
    lettuce["crop_type"]    = "lettuce"
    strawberry["crop_type"] = "strawberry"
    df = pd.concat([lettuce, strawberry], ignore_index=True)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["growth_stage_encoded"] = le.fit_transform(df["growth_stage"].fillna("unknown"))
    df["crop_type_encoded"]    = (df["crop_type"] == "strawberry").astype(int)

    FEATURES = joblib.load("models/growth_model_features.pkl")
    df_clean = df[FEATURES + ["plant_height_cm"]].dropna()
    X = df_clean[FEATURES]
    y = df_clean["plant_height_cm"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load("models/growth_model.pkl")
    preds = model.predict(X_test)
    r2    = r2_score(y_test, preds)
    assert r2 >= 0.80, f"R² too low: {r2:.4f} (need >= 0.80)"
    print(f"Test 3 passed: R² = {r2:.4f} (target >= 0.80)")

def test_charts_saved():
    assert os.path.exists("docs/growth_feature_importance.png"), "Feature importance chart missing!"
    assert os.path.exists("docs/growth_actual_vs_predicted.png"), "Actual vs predicted chart missing!"
    print("Test 4 passed: charts saved correctly")

if __name__ == "__main__":
    test_model_files_exist()
    test_model_predicts_reasonable_height()
    test_model_r2_above_threshold()
    test_charts_saved()
    print("\nAll growth model tests passed!")