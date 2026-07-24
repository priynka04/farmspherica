# Week 8 — SHAP Explainability Report

## What is SHAP?

SHAP (SHapley Additive exPlanations) answers the question: *why* did the model predict this value? Each feature gets a SHAP value — positive means it pushed the prediction UP, negative means it pulled the prediction DOWN. The feature with the largest negative SHAP value is the growth bottleneck.

## Global Feature Importance

*(Average absolute SHAP value across 200 training samples)*

| Rank | Feature | Mean |SHAP| (cm impact) |
|---|---|---|
| 1 | day_after_transplant | 6.7824 |
| 2 | leaf_count | 1.2478 |
| 3 | growth_stage_encoded | 0.5200 |
| 4 | PPFD_umol | 0.1259 |
| 5 | water_temp_C | 0.0874 |
| 6 | pH | 0.0728 |
| 7 | TDS_ppm | 0.0647 |
| 8 | humidity_pct | 0.0646 |
| 9 | photoperiod_hr | 0.0625 |
| 10 | EC_mScm | 0.0591 |
| 11 | ambient_temp_C | 0.0540 |
| 12 | DO_mgL | 0.0491 |
| 13 | crop_type_encoded | 0.0013 |

## Example Predictions

### Healthy plant (Day 30)

- **Predicted height:** 18.97 cm
- **Bottleneck:** none — plant conditions are optimal
- **Top positive drivers:** [('day_after_transplant', np.float64(4.568)), ('growth_stage_encoded', np.float64(0.622)), ('humidity_pct', np.float64(0.21))]
- **Top negative drivers:** [('crop_type_encoded', np.float64(-0.001)), ('pH', np.float64(-0.021)), ('DO_mgL', np.float64(-0.033))]
- **Explanation:** Predicted height: 19.0 cm. Main drivers pushing height UP: day_after_transplant (+4.568cm), growth_stage_encoded (+0.622cm), humidity_pct (+0.21cm). Main bottlenecks pulling height DOWN: crop_type_encoded (-0.001cm), pH (-0.021cm), DO_mgL (-0.033cm). Biggest bottleneck: none — plant conditions are optimal.

### Low EC bottleneck (Day 30)

- **Predicted height:** 18.75 cm
- **Bottleneck:** water_temp_C
- **Top positive drivers:** [('day_after_transplant', np.float64(4.585)), ('growth_stage_encoded', np.float64(0.612)), ('humidity_pct', np.float64(0.206))]
- **Top negative drivers:** [('pH', np.float64(-0.028)), ('DO_mgL', np.float64(-0.034)), ('water_temp_C', np.float64(-0.104))]
- **Explanation:** Predicted height: 18.7 cm. Main drivers pushing height UP: day_after_transplant (+4.585cm), growth_stage_encoded (+0.612cm), humidity_pct (+0.206cm). Main bottlenecks pulling height DOWN: pH (-0.028cm), DO_mgL (-0.034cm), water_temp_C (-0.104cm). Biggest bottleneck: water_temp_C.

### Early stage plant (Day 10)

- **Predicted height:** 2.68 cm
- **Bottleneck:** none — plant conditions are optimal
- **Top positive drivers:** [('photoperiod_hr', np.float64(0.053)), ('growth_stage_encoded', np.float64(0.033)), ('ambient_temp_C', np.float64(0.02))]
- **Top negative drivers:** [('crop_type_encoded', np.float64(-0.0)), ('DO_mgL', np.float64(-0.005)), ('PPFD_umol', np.float64(-0.007))]
- **Explanation:** Predicted height: 2.7 cm. Main drivers pushing height UP: photoperiod_hr (+0.053cm), growth_stage_encoded (+0.033cm), ambient_temp_C (+0.02cm). Main bottlenecks pulling height DOWN: crop_type_encoded (-0.0cm), DO_mgL (-0.005cm), PPFD_umol (-0.007cm). Biggest bottleneck: none — plant conditions are optimal.

## Known Limitations

- SHAP values computed on synthetic training data. Will be re-run on real plant data once available.
- Only the growth prediction model (Random Forest) is explained here. Disease detection (YOLO26n) uses a different architecture where SHAP is not directly applicable.
