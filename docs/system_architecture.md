# System Architecture — Farmspherica Nano PAW

## Overview

The Nano PAW system has 4 services running simultaneously:

| Service | Port | File | What it does |
|---|---|---|---|
| RAG API | 8000 | api/rag_api.py | Answers farming questions from research papers |
| Dashboard API | 8001 | api/dashboard_api.py | Serves sensor data to the frontend |
| Image API | 8002 | api/image_api.py | Handles photo upload and CV prediction |
| Streamlit UI | 8501 | frontend/dashboard.py | The web interface the team uses daily |

## How Data Flows Through the System
Ambika fills NanoPAW_Datasheet.xlsx daily
↓
Livia exports it as CSV
↓
python -c "from api.data_pipeline import run_pipeline; run_pipeline('data/new_data.csv')"
↓
data_pipeline.py validates and stores in data/farmspherica.db (SQLite)
↓
dashboard_api.py (port 8001) reads from database
↓
anomaly_detection.py checks each reading:
Rule-based → CRITICAL if outside safe range
Isolation Forest → WARNING if ML model flags anomaly
↓
frontend/dashboard.py shows live cards, charts, alerts, table
↓
Ambika uploads photo via dashboard
↓
image_api.py saves photo to photos/YYYY-MM-DD/filename.jpg
↓
cv_model.py (EfficientNetB3) auto-predicts condition
↓
Photo + predicted label stored in plant_photos table
↓
Team types question in RAG chat box
↓
rag_api.py → rag_chain.py:
1. Query rewriting (makes question clearer)
2. FAISS searches 9017 chunks for top 10 matches
3. Cohere reranks to top 5
4. Context compression removes irrelevant sentences
5. Groq LLM generates answer with citation
6. Confidence score returned (HIGH/LOW)
↓
Answer shown in dashboard chat with sources cited

## All Files and What They Do

### API Files (api/)
| File | Purpose |
|---|---|
| data_pipeline.py | Reads CSV, validates, stores in SQLite |
| preprocessing.py | Cleans data, normalises, adds rolling averages |
| anomaly_detection.py | Rule-based + Isolation Forest alerts |
| cv_model.py | EfficientNetB3 plant condition classifier |
| document_loader.py | Loads PDFs into text chunks |
| vector_store.py | Builds FAISS index from chunks |
| rag_chain.py | RAG retrieval + Groq LLM + memory + guardrails |
| rag_api.py | FastAPI /ask endpoint |
| dashboard_api.py | FastAPI sensor data endpoints |
| image_api.py | FastAPI photo upload + CV prediction |
| convert_real_data.py | Converts Excel data to CSV format |
| generate_training_data.py | Generates synthetic data for model training |

### Model Files (models/)
| File | Purpose |
|---|---|
| faiss_index/ | FAISS vector index (9017 chunks from research papers) |
| cv_model_v1.h5 | Trained EfficientNetB3 model (100% test accuracy) |
| cv_class_dict.csv | Class index mapping (Healthy=0, Stressed=1, Deficient=2) |
| isolation_forest.pkl | Trained Isolation Forest anomaly model |
| scaler.pkl | StandardScaler used with Isolation Forest |

### Frontend (frontend/)
| File | Purpose |
|---|---|
| dashboard.py | Streamlit web app with 6 sections |

### Documentation (docs/)
| File | Purpose |
|---|---|
| system_architecture.md | This file |
| api_reference.md | All API endpoints documented |
| how_to_run.md | Step-by-step setup and run guide |
| rag_architecture.md | RAG system details |
| dashboard_architecture.md | Dashboard system details |
| schema_notes.md | All 28 sensor column names and valid ranges |
| knowledge_base/ | 3169 pages of hydroponic research PDFs |

### Data Files (data/)
| File | Purpose |
|---|---|
| farmspherica.db | SQLite database (sensor_readings + plant_photos) |
| strawberry_real_data.csv | 7 days of real data from Ambika |
| extended_training_data.csv | 50 rows synthetic data for model training |
| clean_dataset.csv | Preprocessed data ready for ML |
| cv_dataset/ | Plant photos for CV model (Healthy/Stressed/Deficient) |

## Database Tables

### sensor_readings
Stores all sensor data from data_pipeline.py
- date, day_number, plant_id, pH, EC, water_temp_C
- plant_height_cm, leaf_count, condition, observer, remarks

### plant_photos
Stores photo metadata from image_api.py
- filename, date, plant_id, condition
- predicted_condition, prediction_confidence (from CV model)
- angle, notes, uploaded_at

## Safe Sensor Ranges (used for alerts)

| Sensor | Min | Max |
|---|---|---|
| pH | 5.5 | 6.5 |
| EC | 0.8 | 2.0 mS/cm |
| water_temp_C | 18.0 | 26.0 °C |
