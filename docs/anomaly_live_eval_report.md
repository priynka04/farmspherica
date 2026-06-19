# Week 3 — Anomaly Detection LIVE Model Evaluation

Features used: ['pH', 'EC', 'water_temp_C']

Dataset: `data\sensor_anomaly_labeled.csv` (2000 rows, 240 labeled anomalies)

Test set size: 500 rows (25% held out, stratified)

## Overall metrics

- Recall: **0.9167** (target: >= 0.90)
- Precision: **0.7857**
- Target met: **True**

## Recall per anomaly type

| Type | n (test) | Recall |
|---|---|---|
| DO_crash | 7 | 0.286 |
| EC_drop | 9 | 1.0 |
| EC_spike | 9 | 1.0 |
| multi_fault | 6 | 1.0 |
| pH_crash | 7 | 1.0 |
| pH_spike | 5 | 1.0 |
| sensor_stuck | 10 | 1.0 |
| temp_spike | 7 | 1.0 |

## Classification report (test set)

```
              precision    recall  f1-score   support

      normal       0.99      0.97      0.98       440
     anomaly       0.79      0.92      0.85        60

    accuracy                           0.96       500
   macro avg       0.89      0.94      0.91       500
weighted avg       0.96      0.96      0.96       500

```

## Confusion matrix (test set)

```
[[425  15]
 [  5  55]]
```

**Known limitation:** `DO_crash` recall is low because there is no dissolved-oxygen sensor logged in the live database — this anomaly type cannot be reliably caught until a DO reading is added to `sensor_readings`. Flagged as a Month 2 ask for Livia/Akash.
