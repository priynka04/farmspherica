# Week 4 - Month 2: YOLOv8 Disease Detection Model

## 📊 Summary
Successfully trained and deployed a YOLOv8 disease detection model for identifying 5 types of lettuce diseases with 87.6% mAP50 accuracy.

## ✅ Completed Tasks

### 1. Model Training
- **Framework:** YOLOv8n (nano)
- **Dataset:** Roboflow Final-data-set-lettuce-diseases (6,361 images)
- **Hardware:** Tesla T4 GPU (Colab)
- **Training Time:** 1 hour 20 minutes
- **Epochs:** 50

### 2. Performance Results
```
Overall mAP50:  0.876 (87.6%) ✅
Precision:      0.896 (89.6%)
Recall:         0.862 (86.2%)
Average F1:     0.874 ✅

Per-Class Performance:
├─ Downy_mildew_on_lettuce:      F1 = 0.911 (Excellent) ⭐
├─ Lettuce Mosaic Virus:          F1 = 0.891 (Excellent) ⭐
├─ Septoria_Blight_on_lettuce:    F1 = 0.902 (Excellent) ⭐
├─ Bacterial:                     F1 = 0.881 (Good)
└─ Powdery_mildew_on_lettuce:     F1 = 0.783 (Good)
```

### 3. Files Created/Modified
```
✅ models/disease_model_v2.pt              (6.3 MB) — Trained weights
✅ api/cv_model.py                        — YOLOv8 inference engine
✅ api/image_api.py                       — Updated with CV integration
✅ generate_test_image.py                 — Test data generator
✅ docs/week_4_detailed_report.md         — Comprehensive documentation
```

### 4. API Implementation
- **Framework:** FastAPI
- **Port:** 8002
- **Endpoints:**
  - `POST /photos/upload` — Upload photo + auto-predict disease
  - `GET /photos/list` — List all predictions
  - `GET /photos/{id}/image` — Retrieve photo
  - `POST /photos/predict/{id}` — Re-run prediction
  - `DELETE /photos/{id}` — Delete photo
  - `GET /health` — System health check

### 5. Testing & Verification
- ✅ Model loads successfully in <1 second
- ✅ API server runs and responds to requests
- ✅ Photo upload → Auto-prediction working
- ✅ Database persistence verified
- ✅ End-to-end pipeline tested

## 🐛 Complications Faced & Solutions

### Issue #1: CPU vs GPU Training
**Problem:** Initial training used CPU (40 min/epoch — would take 33 hours for 50 epochs)
**Solution:** Enabled Tesla T4 GPU in Colab runtime
**Result:** Reduced to 1.6 min/epoch — saved 30+ hours of training time

### Issue #2: Missing Packages After Session Restart
**Problem:** After enabling GPU, ultralytics and roboflow packages were cleared
**Solution:** Reinstalled: `pip install ultralytics roboflow -q`
**Result:** Training resumed successfully

### Issue #3: Dataset Format Warnings
**Problem:** Dataset contained mixed segmentation/detection data
**Warning:** "Box and segment counts should be equal"
**Solution:** YOLOv8 handled automatically by dropping segments and using boxes
**Result:** No manual intervention needed, model trained successfully

### Issue #4: Windows PowerShell Compatibility
**Problem:** curl commands with `-X POST` syntax failed on PowerShell
**Error:** "A parameter cannot be found that matches parameter name 'X'"
**Solution:** Used Python requests library instead for testing
**Result:** API testing successful on Windows

### Issue #5: Missing Test Data
**Problem:** No plant photos available to test API
**Solution:** Created synthetic test image generator (`generate_test_image.py`)
**Result:** Full pipeline testing completed

## 📈 Performance Analysis

### Best Performing Classes
1. **Septoria_Blight_on_lettuce** — mAP50: 0.974 (97.4%)
2. **Downy_mildew_on_lettuce** — F1: 0.911 (Excellent recall: 0.941)
3. **Lettuce Mosaic Virus** — F1: 0.891 (Excellent recall: 0.935)

### Class Needing Improvement
- **Powdery_mildew_on_lettuce** — Recall: 0.680 (may miss 32% of cases)
  - Recommendation: Collect more training samples or different angles

## 🚀 System Architecture

```
User Uploads Photo
        ↓
api/image_api.py (FastAPI)
        ↓
api/cv_model.py (YOLOv8 Inference)
        ↓
Prediction: {disease, confidence, timestamp}
        ↓
SQLite Database (data/farmspherica.db)
        ↓
REST API Response
```

## 📚 Documentation
- `docs/week_4_detailed_report.md` — Complete technical report
- This file — Week summary

## 🔄 Next Steps (Week 5)
1. Dashboard integration for visual display
2. Batch photo testing with multiple samples
3. Real plant photo validation
4. Performance monitoring and logging
5. Consider retraining Powdery_mildew with additional data
6. Production deployment considerations

## 💾 Model Download
```
Download from Colab:
- Source: /content/runs/detect/farmspherica_disease/v2/weights/best.pt
- Saved as: models/disease_model_v2.pt
- Size: 6.3 MB
- Format: PyTorch (.pt)
```

## 🧪 How to Test

### Start API Server
```bash
uvicorn api.image_api:app --port 8002 --reload
```

### Check System Health
```bash
curl http://localhost:8002/health
```

### Upload Photo & Get Prediction
```python
import requests
files = {'file': open('test_photo.jpg', 'rb')}
data = {'plant_id': 'P01', 'condition': 'Unknown', 'angle': 'Front'}
response = requests.post('http://localhost:8002/photos/upload', files=files, data=data)
print(response.json())
```

## 📊 Metrics Summary
| Metric | Value |
|--------|-------|
| Training Time | 1.33 hours |
| Time per Epoch | 1.6 minutes |
| Model Size | 6.3 MB |
| Overall mAP50 | 87.6% ✅ |
| Average F1 Score | 0.874 ✅ |
| Inference Speed | 2.4ms/image |
| Classes | 5 diseases |
| Training Images | 4,578 |
| Validation Images | 783 |

## 🎯 Objectives Met
- ✅ Dataset downloaded and prepared
- ✅ Model trained to target performance (F1 ≥ 0.85)
- ✅ API integration completed
- ✅ System tested end-to-end
- ✅ Documentation complete
- ✅ Code pushed to GitHub

---

**Status: WEEK 4 COMPLETE** ✅

Ready for Week 5 dashboard integration and real-world testing.