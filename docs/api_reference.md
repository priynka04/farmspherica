# API Reference — Farmspherica Nano PAW

## RAG API — port 8000
Interactive docs: http://localhost:8000/docs

| Method | Endpoint | What it does | Request body |
|---|---|---|---|
| GET | / | Check API is running | none |
| POST | /ask | Ask a farming question | {"question": "your question"} |
| POST | /clear-memory | Reset conversation history | none |
| GET | /health | Health check with version info | none |

### POST /ask — Example response
```json
{
  "question": "What is the ideal pH for hydroponic lettuce?",
  "answer": "The ideal pH range is 5.8–6.2 (Source: fabricius-e-20230831.pdf)...",
  "sources": ["fabricius-e-20230831.pdf", "Nutrient_Solution_for_Hydroponics.pdf"],
  "chunks_retrieved": 5,
  "confidence": "HIGH",
  "rewritten_query": "What is the optimal pH range for growing hydroponic lettuce?"
}
```

---

## Dashboard API — port 8001
Interactive docs: http://localhost:8001/docs

| Method | Endpoint | What it does |
|---|---|---|
| GET | /data/latest | Most recent sensor reading |
| GET | /data/trends | Last 7 readings for trend charts |
| GET | /alerts | Anomaly detection on latest reading |
| GET | /plants | All plant records (full table) |
| GET | /health | Health check |

### GET /alerts — Example response
```json
{
  "alerts": ["pH is OUT OF RANGE: 3.00 (safe range: 5.5 – 6.5)"],
  "count": 1,
  "alert_level": "CRITICAL",
  "ml_result": {
    "is_anomaly": true,
    "confidence": "HIGH",
    "score": -0.1234,
    "message": "ANOMALY DETECTED by ML model"
  },
  "summary": "1 rule alert(s). ML: ANOMALY DETECTED by ML model"
}
```

---

## Image API — port 8002
Interactive docs: http://localhost:8002/docs

| Method | Endpoint | What it does | Parameters |
|---|---|---|---|
| POST | /photos/upload | Upload photo + auto CV prediction | file, plant_id, condition, angle, date, notes |
| GET | /photos/list | List all photos | ?condition=Healthy &plant_id=P01 |
| GET | /photos/{id}/image | Serve the actual image file | photo id |
| POST | /photos/predict/{id} | Re-run CV model on existing photo | photo id |
| DELETE | /photos/{id} | Delete photo from folder and DB | photo id |
| GET | /health | Health check with CV model status | none |

### POST /photos/upload — Example response
```json
{
  "success": true,
  "photo_id": 3,
  "filename": "2026-06-14_P01_Healthy_Front.jpg",
  "cv_prediction": {
    "predicted_class": "Healthy",
    "confidence": 0.994,
    "confidence_pct": "99.4%",
    "all_probabilities": {
      "Healthy": 0.9938,
      "Stressed": 0.0033,
      "Deficient": 0.0030
    },
    "model_used": "EfficientNetB3 fine-tuned"
  },
  "message": "Photo uploaded successfully. CV prediction: Healthy"
}
```

---

## RAG System Details

| Item | Value |
|---|---|
| LLM | Groq llama-3.1-8b-instant (free) |
| Embedding model | HuggingFace all-MiniLM-L6-v2 (local) |
| Reranker | Cohere rerank-english-v3.0 |
| Vector store | FAISS (9017 chunks) |
| Retrieval | FAISS top-10 → Cohere rerank → top-5 |
| RAG features | Query rewriting, context compression, guardrails |
| Accuracy | 10/10 = 100% on test questions |

## CV Model Details

| Item | Value |
|---|---|
| Architecture | EfficientNetB3 (pretrained ImageNet) |
| Classes | Healthy, Stressed, Deficient |
| Training photos | 1200 (400 per class) |
| Train accuracy | 100% |
| Validation accuracy | 100% |
| Test accuracy | 100% |
| Prediction confidence | 99.4% |
| Training time | 30 min 13 sec |

## Anomaly Detection Details

| Item | Value |
|---|---|
| Method 1 | Rule-based (hardcoded safe ranges) |
| Method 2 | Isolation Forest (sklearn, 100 trees) |
| Training rows | 57 rows |
| Features | pH, EC, water_temp_C |
| Anomalies detected | 6/57 rows |
| Model saved | models/isolation_forest.pkl |