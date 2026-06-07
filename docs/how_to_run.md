# How to Run — Farmspherica Nano PAW

## Prerequisites
- Python 3.9 or higher
- Git
- 8GB RAM recommended (minimum 6GB)

## First-time setup

### 1. Clone the repository
```bash
git clone https://github.com/priynka04/farmspherica.git
cd farmspherica
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install all libraries
```bash
pip install fastapi uvicorn langchain langchain-community langchain-groq
pip install langchain-huggingface faiss-cpu sentence-transformers
pip install cohere pandas scikit-learn sqlite3 python-dotenv
pip install streamlit plotly requests pillow
pip install tensorflow
```

### 4. Set up environment variables
Create a `.env` file in the root folder:
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
- Groq key: https://console.groq.com (free)
- Cohere key: https://dashboard.cohere.com (free)

### 5. Rebuild the RAG vector store (first time only)
```bash
python api/vector_store.py
```
Takes about 10 minutes. Only needed once after cloning.

### 6. Retrain models (first time only)
```bash
python api/generate_training_data.py
python -c "from api.anomaly_detection import train_isolation_forest; train_isolation_forest()"
python api/cv_model.py
```

---

## Daily startup — One-click (Windows)

Double-click `start_all.bat` in the farmspherica/ folder.

Then open: **http://localhost:8501**

---

## Daily startup — Manual (4 terminals)

Open 4 terminal windows, activate venv in each, run one command per terminal:

```bash
# Terminal 1 — RAG API (start this FIRST)
uvicorn api.rag_api:app --port 8000

# Terminal 2 — Dashboard API
uvicorn api.dashboard_api:app --reload --port 8001

# Terminal 3 — Image API
uvicorn api.image_api:app --reload --port 8002

# Terminal 4 — Dashboard UI
streamlit run frontend/dashboard.py
```

**Important:** Always start Terminal 1 (RAG API) first, without --reload.
This is because the RAG system loads a large FAISS index and needs to
initialise before TensorFlow (used by Image API) claims RAM.

---

## Adding new sensor data

When Ambika sends a new Excel file:
1. Convert it: `python api/convert_real_data.py`
2. Run pipeline: `python -c "from api.data_pipeline import run_pipeline; run_pipeline('data/strawberry_real_data.csv')"`
3. Refresh dashboard at http://localhost:8501

---

## Retraining models with new data

```bash
# Retrain anomaly detection (after getting 50+ rows of real data)
python -c "from api.anomaly_detection import train_isolation_forest; train_isolation_forest()"

# Retrain CV model (after adding more plant photos)
python api/cv_model.py

# Rebuild RAG vector store (after adding new research papers)
python api/vector_store.py
```

---

## Running all tests

```bash
python tests/test_pipeline.py
python tests/test_rag.py
python tests/test_dashboard.py
python tests/test_anomaly.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| MemoryError on startup | Start RAG API first without --reload, then other services |
| Dashboard shows empty table | Check API is running on port 8001 |
| Photo upload fails | Make sure Image API is running on port 8002 |
| RAG chat not working | Make sure RAG API is running on port 8000 |
| CV model not predicting | Run python api/cv_model.py to train first |
| FAISS index not found | Run python api/vector_store.py to rebuild |