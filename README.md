# Farmspherica Nano PAW — Data Pipeline

## What this does
Reads hydroponic sensor data from CSV files, validates it, stores it in a SQLite database,
and produces a clean preprocessed dataset for AI/ML model training.

## How to run
1. Activate virtual environment: `source venv/bin/activate`
2. Run pipeline: `python api/data_pipeline.py`
3. Run preprocessing: `python api/preprocessing.py`
4. Run tests: `python tests/test_pipeline.py`

## Folder structure
- data/         → raw and processed CSV files, SQLite database
- api/          → pipeline and preprocessing scripts
- tests/        → unit tests
- docs/         → documentation and diagrams
- models/       → (for Month 2 ML models)
- frontend/     → (for Month 2 dashboard)

## Week 2 — RAG Smart Farming Assistant

### How to run the RAG system
1. Build vector store (first time only): `python api/vector_store.py`
2. Test RAG chain: `python api/rag_chain.py`
3. Start API: `uvicorn api.rag_api:app --reload --port 8000`
4. Run tests: `python tests/test_rag.py`
5. Interactive docs: http://localhost:8000/docs

### RAG Stats
- 3169 pages loaded from 5 topic folders
- 9014 chunks indexed in FAISS
- LLM: Groq llama-3.3-70b-versatile (free)
- Embeddings: HuggingFace all-MiniLM-L6-v2 (free, local)