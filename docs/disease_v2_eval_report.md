# YOLOv8n Lettuce Disease Detection — v2 Evaluation Report

## Overall Metrics
- **mAP50:** 0.876
- **Precision:** 0.896
- **Recall:** 0.862
- **mAP50-95:** 0.754
- **Training Time:** 1.326 hours (1h 20m)
- **Dataset:** Final-data-set-lettuce-diseases-1 (4,578 train | 783 val)
- **Model:** YOLOv8n (pretrained)

## Per-Class Performance

| Class | Precision | Recall | mAP50 | F1 Score |
|-------|-----------|--------|-------|----------|
| Bacterial | 0.930 | 0.838 | 0.842 | 0.881 |
| Downy_mildew_on_lettuce | 0.882 | 0.941 | 0.927 | 0.911 |
| Lettuce Mosaic Virus | 0.853 | 0.935 | 0.888 | 0.891 |
| Powdery_mildew_on_lettuce | 0.923 | 0.680 | 0.750 | 0.783 |
| Septoria_Blight_on_lettuce | 0.889 | 0.916 | 0.974 | 0.902 |

**Average F1 Score: 0.874** ✅

## Model Location
- Weights: `/content/runs/detect/farmspherica_disease/v2/weights/best.pt`
- Saved locally as: `models/disease_model_v2.pt`

## Next Steps
1. Save this report to `docs/disease_v2_eval_report.md`
2. Provide `api/image_api.py` for integration into dashboard
