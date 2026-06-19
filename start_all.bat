@echo off
echo Starting Farmspherica Nano PAW System...
start "RAG API" cmd /k "cd /d C:\Users\Lenovo\farmspherica && venv\Scripts\activate && uvicorn api.rag_api:app --port 8000"
timeout /t 5
start "Dashboard API" cmd /k "cd /d C:\Users\Lenovo\farmspherica && venv\Scripts\activate && uvicorn api.dashboard_api:app --reload --port 8001"
timeout /t 3
start "Image API" cmd /k "cd /d C:\Users\Lenovo\farmspherica && venv\Scripts\activate && uvicorn api.image_api:app --reload --port 8002"
timeout /t 3
start "Dashboard" cmd /k "cd /d C:\Users\Lenovo\farmspherica && venv\Scripts\activate && streamlit run frontend/dashboard.py"
echo All services started! Open: http://localhost:8501 
