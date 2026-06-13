# =============================================================
# FILE: api/lstm_growth.py
# WHAT IT DOES: Trains an LSTM to forecast the next 7 days
#               of plant height growth (trajectory, not a point).
#               Sequences are built PER PLANT — no cross-plant
#               data leakage within a single sequence.
#               Train/test split is RANDOM 80/20 across all
#               sequences so both sets cover all growth phases.
#               Delta features (day-over-day changes) added to
#               give the model explicit growth-rate signal.
# HOW TO RUN:   python api/lstm_growth.py
# REQUIRES:     pip install tensorflow
# =============================================================

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.makedirs("models", exist_ok=True)
os.makedirs("docs",   exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)

# =============================================================
# STEP 1 — Load and combine datasets
# =============================================================

print("[INFO] Loading datasets...")
lettuce    = pd.read_csv("data/lollo_rosa_lettuce_synthetic.csv")
strawberry = pd.read_csv("data/strawberry_synthetic.csv")

lettuce["crop_type_encoded"]    = 0
strawberry["crop_type_encoded"] = 1

lettuce["plant_id"]    = "L_" + lettuce["plant_id"].astype(str)
strawberry["plant_id"] = "S_" + strawberry["plant_id"].astype(str)

df = pd.concat([lettuce, strawberry], ignore_index=True)
print(f"[INFO] Combined rows : {len(df)}")
print(f"[INFO] Unique plants : {df['plant_id'].nunique()}")

# =============================================================
# STEP 2 — Add delta features PER PLANT before any splitting.
#           delta_height  = height(t) - height(t-1)
#           delta_leaves  = leaf_count(t) - leaf_count(t-1)
#           Day 1 of each plant gets delta = 0 (no prior day).
#           These give the model explicit growth-rate signal so
#           it doesn't have to infer velocity from raw values.
# =============================================================

cols_base = [
    "day_after_transplant",
    "water_temp_C",
    "ambient_temp_C",
    "pH",
    "EC_mScm",
    "DO_mgL",
    "humidity_pct",
    "PPFD_umol",
    "leaf_count",
    "crop_type_encoded",
]
TARGET = "plant_height_cm"

cols_needed = ["plant_id"] + cols_base + [TARGET]
df_clean = df[cols_needed].dropna()
df_clean = df_clean.sort_values(["plant_id", "day_after_transplant"]).reset_index(drop=True)

# Compute deltas within each plant (no cross-plant bleeding)
df_clean["delta_height"] = (
    df_clean.groupby("plant_id")["plant_height_cm"]
    .diff()
    .fillna(0)
)
df_clean["delta_leaves"] = (
    df_clean.groupby("plant_id")["leaf_count"]
    .diff()
    .fillna(0)
)

FEATURES = cols_base + ["delta_height", "delta_leaves"]   # 12 features

print(f"[INFO] Clean rows    : {len(df_clean)}")
print(f"[INFO] Features      : {len(FEATURES)}  {FEATURES}")

# =============================================================
# STEP 3 — Fit scalers on full dataset
# =============================================================

feature_scaler = MinMaxScaler()
target_scaler  = MinMaxScaler()

feature_scaler.fit(df_clean[FEATURES])
target_scaler.fit(df_clean[[TARGET]])

# =============================================================
# STEP 4 — Build ALL sequences per plant, collect into one pool
# =============================================================

LOOK_BACK = 10
FORECAST  = 7

def build_plant_sequences(plant_df, look_back, forecast):
    X_raw = feature_scaler.transform(plant_df[FEATURES])
    y_raw = target_scaler.transform(plant_df[[TARGET]])
    Xs, ys = [], []
    for i in range(len(X_raw) - look_back - forecast + 1):
        Xs.append(X_raw[i : i + look_back])
        ys.append(y_raw[i + look_back : i + look_back + forecast, 0])
    return Xs, ys

all_X, all_y = [], []

for plant_id, group in df_clean.groupby("plant_id"):
    group = group.sort_values("day_after_transplant")
    Xs, ys = build_plant_sequences(group, LOOK_BACK, FORECAST)
    all_X.extend(Xs)
    all_y.extend(ys)

X_seq = np.array(all_X)
y_seq = np.array(all_y)
print(f"[INFO] Total sequences : {len(X_seq)}  shape: {X_seq.shape}")

# =============================================================
# STEP 5 — Random 80/20 train/test split across all sequences
# =============================================================

indices = np.arange(len(X_seq))
np.random.shuffle(indices)
split   = int(len(indices) * 0.80)

train_idx = indices[:split]
test_idx  = indices[split:]

X_train, y_train = X_seq[train_idx], y_seq[train_idx]
X_test,  y_test  = X_seq[test_idx],  y_seq[test_idx]

print(f"[INFO] Train sequences : {len(X_train)}  shape: {X_train.shape}")
print(f"[INFO] Test  sequences : {len(X_test)}   shape: {X_test.shape}")

# =============================================================
# STEP 6 — Build LSTM model
# =============================================================

model = Sequential([
    LSTM(96, return_sequences=True,
         input_shape=(LOOK_BACK, len(FEATURES))),
    Dropout(0.3),
    LSTM(48, return_sequences=False),
    Dropout(0.2),
    Dense(24, activation="relu"),
    Dense(FORECAST)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="huber"
)
model.summary()

# =============================================================
# STEP 7 — Train
# =============================================================

early_stop = EarlyStopping(
    monitor="val_loss", patience=25,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5,
    patience=8, min_lr=1e-6, verbose=1
)

print("\n[INFO] Training LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =============================================================
# STEP 8 — Evaluate
# =============================================================

y_pred_scaled = model.predict(X_test)
y_pred_cm     = target_scaler.inverse_transform(y_pred_scaled)
y_test_cm     = target_scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_cm.flatten(), y_pred_cm.flatten())
print(f"\n  LSTM — MAE: {mae:.4f} cm  (target: < 1.5 cm)")

# =============================================================
# STEP 9 — Save everything
# =============================================================

model.save("models/lstm_growth.keras")
joblib.dump(feature_scaler, "models/lstm_feature_scaler.pkl")
joblib.dump(target_scaler,  "models/lstm_target_scaler.pkl")
joblib.dump(FEATURES,       "models/lstm_features.pkl")
joblib.dump(LOOK_BACK,      "models/lstm_lookback.pkl")
joblib.dump(test_idx,       "models/lstm_test_idx.pkl")
print("  LSTM saved → models/lstm_growth.keras")

# =============================================================
# STEP 10 — Training loss chart
# =============================================================

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"],     label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Huber Loss")
plt.title("LSTM Training Loss (random 80/20 + delta features)")
plt.legend()
plt.tight_layout()
plt.savefig("docs/lstm_training_loss.png", dpi=150)
plt.close()
print("  Chart saved → docs/lstm_training_loss.png")

# =============================================================
# STEP 11 — Sample trajectory plots (6 test sequences)
# =============================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes      = axes.flatten()
sample_indices = np.linspace(0, len(X_test) - 1, 6, dtype=int)

for i, idx in enumerate(sample_indices):
    actual    = y_test_cm[idx]
    predicted = y_pred_cm[idx]
    days      = np.arange(1, FORECAST + 1)
    axes[i].plot(days, actual,    "b-o",  label="Actual",    markersize=4)
    axes[i].plot(days, predicted, "r--s", label="Predicted", markersize=4)
    axes[i].set_title(f"Test sequence {i+1}")
    axes[i].set_xlabel("Day ahead")
    axes[i].set_ylabel("Height (cm)")
    axes[i].legend(fontsize=7)

plt.suptitle("LSTM — 7-Day Growth Trajectory (delta features)", fontsize=13)
plt.tight_layout()
plt.savefig("docs/lstm_trajectory_samples.png", dpi=150)
plt.close()
print("  Chart saved → docs/lstm_trajectory_samples.png")

# =============================================================
# SUMMARY
# =============================================================

print(f"""
{'='*50}
  WEEK 2 COMPLETE — LSTM Growth Trajectory
  Split       : random 80/20 across all sequences
  Features    : {len(FEATURES)} (added delta_height, delta_leaves)
  MAE         : {mae:.4f} cm  (target: < 1.5 cm)
  Look-back   : {LOOK_BACK} days
  Forecast    : {FORECAST} days ahead
  Train seqs  : {len(X_train)}
  Test  seqs  : {len(X_test)}
  Saved to    : models/lstm_growth.keras
{'='*50}
""")