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