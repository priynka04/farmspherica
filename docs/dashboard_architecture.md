# Dashboard Architecture — Farmspherica Nano PAW

## What it does
A live web dashboard that shows real hydroponic sensor data, trend charts,
automated alerts, plant photos, and a built-in AI chat assistant.

## How it works
The dashboard has three separate services running at the same time:

1. RAG API (port 8000) — answers farming questions using research papers
2. Dashboard API (port 8001) — reads sensor data from SQLite and serves it
3. Image API (port 8002) — handles photo uploads and the photo gallery
4. Streamlit frontend (port 8501) — the visual web interface the team sees

## Files
- api/dashboard_api.py  → FastAPI backend serving sensor data endpoints
- api/image_api.py      → FastAPI backend for photo upload and gallery
- frontend/dashboard.py → Streamlit frontend (the actual web page)
- photos/               → folder where all uploaded plant photos are stored

## API Endpoints

### Dashboard API (port 8001)
- GET /data/latest → returns the most recent sensor reading
- GET /data/trends → returns last 7 readings for trend charts
- GET /alerts      → checks if any sensor is outside safe range
- GET /plants      → returns all plant records (full table)
- GET /health      → confirms the API is running

### Image API (port 8002)
- POST /photos/upload  → upload a photo with plant_id, condition, angle
- GET  /photos/list    → list all photos (filter by condition or plant_id)
- DELETE /photos/{id}  → delete a photo from folder and database

## Safe sensor ranges (used for alerts)
- pH:           4.0 – 9.0
- EC:           0.0 – 5.0 mS/cm
- Water temp:   10 – 35°C

## Dashboard sections
1. Alerts panel    — red boxes appear when any sensor is out of range
2. Sensor cards    — latest pH, EC, temperature, height, leaf count
3. Trend charts    — pH and EC line charts, height bar chart, temp chart
4. Plant table     — full history of all 7 days of strawberry data
5. Photo logger    — upload tab + gallery tab with 3-column grid view
6. RAG chat        — type any farming question, get answer with sources

## How to run everything
Open 4 terminal windows and run one command in each:

Terminal 1: uvicorn api.rag_api:app --reload --port 8000
Terminal 2: uvicorn api.dashboard_api:app --reload --port 8001
Terminal 3: uvicorn api.image_api:app --reload --port 8002
Terminal 4: streamlit run frontend/dashboard.py

Then open: http://localhost:8501

## Database tables
- sensor_readings  → all sensor data from data_pipeline.py
- plant_photos     → metadata for all uploaded photos (created by image_api.py)

## Test results
| Test                              | Result  |
|-----------------------------------|---------|
| Database has data                 | PASSED  |
| Alert logic catches out-of-range  | PASSED  |
| plant_photos table exists         | PASSED  |
| Trends returns max 7 rows         | PASSED  |