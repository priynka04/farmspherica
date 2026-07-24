"""
api/shap_explainer.py
Week 8 — SHAP Explainability for the Growth Prediction Model

Answers the question: "Why is the model predicting this plant height?
Which sensor reading is the biggest bottleneck?"

Uses TreeExplainer (fast, exact SHAP values for tree-based models like
Random Forest) on the growth model trained in Week 5 of Month 2.

Run directly to generate the full SHAP explanation report:
    python api/shap_explainer.py
"""

import joblib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROWTH_MODEL_PATH = Path("models/growth_model.pkl")
REPORT_PATH       = Path("docs/shap_explanation_report.md")

FEATURE_COLS = [
    "day_after_transplant", "water_temp_C", "ambient_temp_C",
    "pH", "EC_mScm", "TDS_ppm", "DO_mgL", "humidity_pct",
    "photoperiod_hr", "PPFD_umol", "leaf_count",
    "growth_stage_encoded", "crop_type_encoded",
]


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_growth_model():
    try:
        model = joblib.load(GROWTH_MODEL_PATH)
    except Exception:
        with open(GROWTH_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    return model


# ---------------------------------------------------------------------------
# Explain a single reading
# ---------------------------------------------------------------------------
def explain_prediction(reading: dict) -> dict:
    """
    Given current sensor readings, returns a plain-English explanation
    of why the model predicted that plant height.

    reading: dict with keys matching FEATURE_COLS.
             Missing keys are filled with typical values.

    Returns dict:
        predicted_height_cm  float
        top_positive         list of (feature, shap_value) — pushing height UP
        top_negative         list of (feature, shap_value) — pulling height DOWN
        bottleneck           str  — the single biggest negative driver
        explanation          str  — human-readable summary
    """
    model = load_growth_model()

    # Fill missing features with reasonable defaults
    defaults = {
        "day_after_transplant": 30,
        "water_temp_C": 20.0, "ambient_temp_C": 23.0,
        "pH": 6.0, "EC_mScm": 1.5, "TDS_ppm": 1000,
        "DO_mgL": 6.8, "humidity_pct": 60.0,
        "photoperiod_hr": 16.0, "PPFD_umol": 250.0,
        "leaf_count": 8, "growth_stage_encoded": 1,
        "crop_type_encoded": 0,
    }
    row = {**defaults, **{k: v for k, v in reading.items() if k in FEATURE_COLS}}
    X   = pd.DataFrame([row])[FEATURE_COLS]

    predicted_height = float(model.predict(X)[0])

    # SHAP values
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]   # shape: (n_features,)

    pairs = sorted(
        zip(FEATURE_COLS, shap_values),
        key=lambda x: x[1], reverse=True
    )

    top_positive = [(f, round(v, 3)) for f, v in pairs if v > 0][:3]
    top_negative = [(f, round(v, 3)) for f, v in pairs if v < 0][:3]
    meaningful_negatives = [(f, v) for f, v in top_negative if abs(v) >= 0.05]
    bottleneck = meaningful_negatives[0][0] if meaningful_negatives else "none — plant conditions are optimal"

    pos_str = ", ".join(
        f"{f} (+{v}cm)" for f, v in top_positive
    ) or "none"
    neg_str = ", ".join(
        f"{f} ({v}cm)" for f, v in top_negative
    ) or "none"

    explanation = (
        f"Predicted height: {predicted_height:.1f} cm. "
        f"Main drivers pushing height UP: {pos_str}. "
        f"Main bottlenecks pulling height DOWN: {neg_str}. "
        f"Biggest bottleneck: {bottleneck}."
    )

    return {
        "predicted_height_cm": round(predicted_height, 2),
        "top_positive":        top_positive,
        "top_negative":        top_negative,
        "bottleneck":          bottleneck,
        "explanation":         explanation,
        "shap_values":         dict(zip(FEATURE_COLS,
                                        [round(v, 4) for v in shap_values])),
    }


# ---------------------------------------------------------------------------
# Global feature importance (average |SHAP| over a dataset)
# ---------------------------------------------------------------------------
def global_feature_importance(n_samples: int = 200) -> pd.DataFrame:
    """
    Computes mean absolute SHAP value per feature over n_samples rows
    from the growth datasets. Higher = more important on average.
    """
    model = load_growth_model()

    # Load a sample of training data
    data_files = [
        Path("data/lollo_rosa_lettuce_synthetic.csv"),
        Path("data/strawberry_synthetic.csv"),
    ]
    dfs = []
    for p in data_files:
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError(
            "Growth dataset CSVs not found in data/. "
            "Make sure lollo_rosa_lettuce_synthetic.csv and "
            "strawberry_synthetic.csv are present."
        )
    df = pd.concat(dfs, ignore_index=True)

    # Keep only rows that have all feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available].dropna()

    # Sample for speed
    sample = df.sample(min(n_samples, len(df)), random_state=42)

    # Fill any missing columns with defaults
    for col in FEATURE_COLS:
        if col not in sample.columns:
            sample[col] = 0
    X = sample[FEATURE_COLS]

    explainer   = shap.TreeExplainer(model)
    shap_matrix = explainer.shap_values(X)   # shape: (n_samples, n_features)

    importance = pd.DataFrame({
        "feature":         FEATURE_COLS,
        "mean_abs_shap":   np.abs(shap_matrix).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance


# ---------------------------------------------------------------------------
# Main — generate full explanation report
# ---------------------------------------------------------------------------
def main():
    print("[INFO] Loading growth model...")
    model = load_growth_model()
    print(f"  Model: {type(model).__name__}")

    print("\n[INFO] Computing global feature importance...")
    importance = global_feature_importance(n_samples=200)
    print("\nTop 5 most important features for growth prediction:")
    print(importance.head(5).to_string(index=False))

    print("\n[INFO] Running example predictions with SHAP explanations...")
    examples = [
        {
            "name": "Healthy plant (Day 30)",
            "reading": {
                "day_after_transplant": 30, "pH": 6.0, "EC_mScm": 1.5,
                "leaf_count": 10, "water_temp_C": 20.0,
                "growth_stage_encoded": 1, "crop_type_encoded": 0,
            }
        },
        {
            "name": "Low EC bottleneck (Day 30)",
            "reading": {
                "day_after_transplant": 30, "pH": 6.0, "EC_mScm": 0.5,
                "leaf_count": 6, "water_temp_C": 20.0,
                "growth_stage_encoded": 1, "crop_type_encoded": 0,
            }
        },
        {
            "name": "Early stage plant (Day 10)",
            "reading": {
                "day_after_transplant": 10, "pH": 6.0, "EC_mScm": 1.5,
                "leaf_count": 4, "water_temp_C": 20.0,
                "growth_stage_encoded": 0, "crop_type_encoded": 0,
            }
        },
    ]

    results = []
    for ex in examples:
        r = explain_prediction(ex["reading"])
        results.append((ex["name"], r))
        print(f"\n  {ex['name']}")
        print(f"    Predicted height: {r['predicted_height_cm']} cm")
        print(f"    Bottleneck:       {r['bottleneck']}")
        print(f"    Explanation:      {r['explanation']}")

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Week 8 — SHAP Explainability Report\n\n")
        f.write("## What is SHAP?\n\n")
        f.write(
            "SHAP (SHapley Additive exPlanations) answers the question: "
            "*why* did the model predict this value? Each feature gets a "
            "SHAP value — positive means it pushed the prediction UP, "
            "negative means it pulled the prediction DOWN. The feature with "
            "the largest negative SHAP value is the growth bottleneck.\n\n"
        )
        f.write("## Global Feature Importance\n\n")
        f.write("*(Average absolute SHAP value across 200 training samples)*\n\n")
        f.write("| Rank | Feature | Mean |SHAP| (cm impact) |\n|---|---|---|\n")
        for i, row in importance.iterrows():
            f.write(f"| {i+1} | {row['feature']} | "
                    f"{row['mean_abs_shap']:.4f} |\n")
        f.write("\n## Example Predictions\n\n")
        for name, r in results:
            f.write(f"### {name}\n\n")
            f.write(f"- **Predicted height:** {r['predicted_height_cm']} cm\n")
            f.write(f"- **Bottleneck:** {r['bottleneck']}\n")
            f.write(f"- **Top positive drivers:** "
                    f"{r['top_positive']}\n")
            f.write(f"- **Top negative drivers:** "
                    f"{r['top_negative']}\n")
            f.write(f"- **Explanation:** {r['explanation']}\n\n")
        f.write("## Known Limitations\n\n")
        f.write(
            "- SHAP values computed on synthetic training data. "
            "Will be re-run on real plant data once available.\n"
            "- Only the growth prediction model (Random Forest) is explained "
            "here. Disease detection (YOLO26n) uses a different architecture "
            "where SHAP is not directly applicable.\n"
        )

    print(f"\nReport saved -> {REPORT_PATH}")
    print("\nWEEK 8 SHAP EXPLAINABILITY — DONE")
    print(f"  Top feature: {importance.iloc[0]['feature']} "
          f"(mean |SHAP|={importance.iloc[0]['mean_abs_shap']:.4f})")
    print(f"  Bottleneck in low-EC example: "
          f"{results[1][1]['bottleneck']}")


if __name__ == "__main__":
    main()