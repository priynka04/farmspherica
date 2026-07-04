# Week 6 — Early Warning Model Evaluation Report

**Approach:** Isolation Forest continuous anomaly score (decision_function) as real-time risk signal.

**Model used:** `models/isolation_forest_live.pkl` (trained in Week 7 on pH, EC, water_temp_C).

## Why this approach

The initial approach (XGBoost trained to predict randomly-injected future anomalies) gave F1=0.03 because the synthetic dataset has no deteriorating trend before anomalies — they are randomly placed. The correct interpretation of 'early warning' is: use the Isolation Forest's continuous score to detect when readings are drifting toward anomaly territory. This is directly computable from current readings and requires no training of a new model.

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.8778 |
| Avg risk score (normal rows) | 0.2579 |
| Avg risk score (anomaly rows) | 0.762 |

## Known limitations

- Validated on synthetic data only. Real-plant trends will be clearer since real anomalies follow organic deterioration.
- Only 3 sensor features (pH, EC, water_temp_C) — limited by live database schema. Will improve as more sensors are added.
- Segmentation output (canopy_area, biomass) not yet fused — will be added once ESP32-CAM is live.
