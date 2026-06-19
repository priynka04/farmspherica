# Week 3 — Anomaly Detection FULL Model Evaluation

Features used: ['pH', 'EC_mScm', 'TDS_ppm', 'water_temp_C', 'DO_mgL', 'ambient_temp_C', 'humidity_pct']

Dataset: `data\sensor_anomaly_labeled.csv` (2000 rows, 240 labeled anomalies)

Test set size: 500 rows (25% held out, stratified)

## Overall metrics

- Recall: **0.9667** (target: >= 0.90)
- Precision: **0.9206**
- Target met: **True**

## Recall per anomaly type

| Type | n (test) | Recall |
|---|---|---|
| DO_crash | 7 | 1.0 |
| EC_drop | 9 | 1.0 |
| EC_spike | 9 | 1.0 |
| multi_fault | 6 | 1.0 |
| pH_crash | 7 | 0.714 |
| pH_spike | 5 | 1.0 |
| sensor_stuck | 10 | 1.0 |
| temp_spike | 7 | 1.0 |

## Classification report (test set)

```
              precision    recall  f1-score   support

      normal       1.00      0.99      0.99       440
     anomaly       0.92      0.97      0.94        60

    accuracy                           0.99       500
   macro avg       0.96      0.98      0.97       500
weighted avg       0.99      0.99      0.99       500

```

## Confusion matrix (test set)

```
[[435   5]
 [  2  58]]
```
