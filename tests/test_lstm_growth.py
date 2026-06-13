import sys
sys.path.append(".")
import os
import numpy as np
import joblib


def test_lstm_files_exist():
    assert os.path.exists("models/lstm_growth.keras"),        "LSTM model missing!"
    assert os.path.exists("models/lstm_feature_scaler.pkl"),  "Feature scaler missing!"
    assert os.path.exists("models/lstm_target_scaler.pkl"),   "Target scaler missing!"
    assert os.path.exists("models/lstm_features.pkl"),        "Features list missing!"
    assert os.path.exists("models/lstm_lookback.pkl"),        "Lookback config missing!"
    assert os.path.exists("models/lstm_test_idx.pkl"),        "Test indices missing!"
    print("Test 1 passed: LSTM files exist")


def test_lstm_predicts_7_days():
    import tensorflow as tf

    model          = tf.keras.models.load_model("models/lstm_growth.keras")
    feature_scaler = joblib.load("models/lstm_feature_scaler.pkl")
    target_scaler  = joblib.load("models/lstm_target_scaler.pkl")
    FEATURES       = joblib.load("models/lstm_features.pkl")
    LOOK_BACK      = joblib.load("models/lstm_lookback.pkl")

    # 12 values: original 10 + delta_height=0, delta_leaves=0
    sample = np.array([[
        d, 20.0, 21.0, 6.0, 1.2, 7.0, 65.0, 220.0, 8, 0, 0.0, 0.0
    ] for d in range(1, LOOK_BACK + 1)])

    sample_scaled = feature_scaler.transform(sample)
    X             = sample_scaled.reshape(1, LOOK_BACK, len(FEATURES))
    pred_scaled   = model.predict(X)
    pred_cm       = target_scaler.inverse_transform(pred_scaled)[0]

    assert len(pred_cm) == 7, f"Expected 7 predictions, got {len(pred_cm)}"
    assert all(0 < p < 60 for p in pred_cm), f"Predictions out of range: {pred_cm}"
    print(f"Test 2 passed: 7-day forecast = {[round(float(p), 1) for p in pred_cm]} cm")


def test_lstm_mae_reasonable():
    import tensorflow as tf
    import pandas as pd
    from sklearn.metrics import mean_absolute_error

    model          = tf.keras.models.load_model("models/lstm_growth.keras")
    feature_scaler = joblib.load("models/lstm_feature_scaler.pkl")
    target_scaler  = joblib.load("models/lstm_target_scaler.pkl")
    FEATURES       = joblib.load("models/lstm_features.pkl")
    LOOK_BACK      = joblib.load("models/lstm_lookback.pkl")
    test_idx       = joblib.load("models/lstm_test_idx.pkl")
    FORECAST       = 7

    lettuce    = pd.read_csv("data/lollo_rosa_lettuce_synthetic.csv")
    strawberry = pd.read_csv("data/strawberry_synthetic.csv")
    lettuce["crop_type_encoded"]    = 0
    strawberry["crop_type_encoded"] = 1
    lettuce["plant_id"]    = "L_" + lettuce["plant_id"].astype(str)
    strawberry["plant_id"] = "S_" + strawberry["plant_id"].astype(str)

    df = pd.concat([lettuce, strawberry], ignore_index=True)

    # Must compute delta features before selecting columns —
    # they don't exist in the raw CSVs, only after this groupby diff.
    df = df.sort_values(["plant_id", "day_after_transplant"]).reset_index(drop=True)
    df["delta_height"] = df.groupby("plant_id")["plant_height_cm"].diff().fillna(0)
    df["delta_leaves"] = df.groupby("plant_id")["leaf_count"].diff().fillna(0)

    cols_needed = ["plant_id"] + FEATURES + ["plant_height_cm"]
    df_clean = df[cols_needed].dropna()
    df_clean = df_clean.sort_values(["plant_id", "day_after_transplant"]).reset_index(drop=True)

    # Rebuild the full sequence pool in the same order as training
    all_X, all_y = [], []
    for plant_id, group in df_clean.groupby("plant_id"):
        group = group.sort_values("day_after_transplant")
        X_raw = feature_scaler.transform(group[FEATURES])
        y_raw = target_scaler.transform(group[["plant_height_cm"]])
        for i in range(len(X_raw) - LOOK_BACK - FORECAST + 1):
            all_X.append(X_raw[i : i + LOOK_BACK])
            all_y.append(y_raw[i + LOOK_BACK : i + LOOK_BACK + FORECAST, 0])

    X_seq = np.array(all_X)
    y_seq = np.array(all_y)

    # Use the exact same held-out indices saved during training
    X_test = X_seq[test_idx]
    y_test = y_seq[test_idx]

    y_pred_scaled = model.predict(X_test)
    y_pred_cm     = target_scaler.inverse_transform(y_pred_scaled)
    y_test_cm     = target_scaler.inverse_transform(y_test)
    mae = mean_absolute_error(y_test_cm.flatten(), y_pred_cm.flatten())

    assert mae < 5.0, f"MAE too high: {mae:.4f} cm (must be < 5.0)"
    if mae < 1.5:
        print(f"Test 3 passed: MAE = {mae:.4f} cm ✅  (target < 1.5 cm — HIT!)")
    else:
        print(f"Test 3 passed: MAE = {mae:.4f} cm ⚠️   (above 1.5 cm target, but < 5.0 — acceptable)")


def test_charts_exist():
    assert os.path.exists("docs/lstm_training_loss.png"),      "Loss chart missing!"
    assert os.path.exists("docs/lstm_trajectory_samples.png"), "Trajectory chart missing!"
    print("Test 4 passed: LSTM charts saved")


if __name__ == "__main__":
    test_lstm_files_exist()
    test_lstm_predicts_7_days()
    test_lstm_mae_reasonable()
    test_charts_exist()
    print("\nAll LSTM tests passed!")