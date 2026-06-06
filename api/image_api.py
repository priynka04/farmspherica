# =============================================================
# FILE: api/image_api.py  (UPDATED — Week 4 version)
# WHAT CHANGED FROM WEEK 3:
#   - Photo upload now automatically runs CV model prediction
#   - Predicted condition is stored in the database
#   - New endpoint: GET /photos/predict/{id} to re-run prediction
# HOW TO RUN:   uvicorn api.image_api:app --port 8002 --reload
# =============================================================

import os
import sqlite3
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import our CV model prediction function from Week 4
# If cv_model is not available yet, we fall back gracefully
try:
    from api.cv_model import predict_plant_condition
    CV_MODEL_AVAILABLE = True
except ImportError:
    CV_MODEL_AVAILABLE = False
    print("[WARNING] cv_model.py not found. Auto-prediction disabled.")

app = FastAPI(title="Farmspherica Image API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH     = "data/farmspherica.db"
PHOTOS_DIR  = "photos"
os.makedirs(PHOTOS_DIR, exist_ok=True)


# =============================================================
# DATABASE SETUP
# =============================================================

def init_db():
    """Create the plant_photos table if it doesn't exist."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plant_photos (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            filename         TEXT NOT NULL,
            date             TEXT,
            plant_id         TEXT,
            condition        TEXT,
            predicted_condition TEXT,
            prediction_confidence REAL,
            angle            TEXT,
            notes            TEXT,
            uploaded_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()


# =============================================================
# ENDPOINTS
# =============================================================

@app.post("/photos/upload")
async def upload_photo(
    file:      UploadFile = File(...),
    plant_id:  str        = Form(default="P01"),
    condition: str        = Form(default="Unknown"),
    angle:     str        = Form(default="Front"),
    date:      str        = Form(default=""),
    notes:     str        = Form(default="")
):
    """
    Upload a plant photo.
    - Saves the file to the photos/ folder
    - Runs CV model to auto-predict the condition
    - Stores everything in SQLite
    - Returns the upload result + CV prediction
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    # Build a clean standardised filename
    # Format: YYYY-MM-DD_PlantID_Condition_Angle.jpg
    ext      = os.path.splitext(file.filename)[1].lower() or ".jpg"
    filename = f"{date}_{plant_id}_{condition}_{angle}{ext}"
    filepath = os.path.join(PHOTOS_DIR, filename)

    # Save file to disk
    with open(filepath, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # --- Run CV model auto-prediction ---
    predicted_condition  = None
    prediction_confidence = None
    cv_result            = {}

    if CV_MODEL_AVAILABLE:
        try:
            cv_result             = predict_plant_condition(filepath)
            predicted_condition   = cv_result.get("predicted_class")
            prediction_confidence = cv_result.get("confidence")
            print(f"[CV] Predicted: {predicted_condition} ({cv_result.get('confidence_pct')})")
        except Exception as e:
            print(f"[CV WARNING] Prediction failed: {e}")
            cv_result = {"error": str(e)}
    else:
        cv_result = {"message": "CV model not available yet. Train it first."}

    # --- Save to database ---
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO plant_photos
            (filename, date, plant_id, condition, predicted_condition,
             prediction_confidence, angle, notes, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename, date, plant_id, condition,
        predicted_condition, prediction_confidence,
        angle, notes, datetime.now().isoformat()
    ))
    conn.commit()
    photo_id = cursor.lastrowid
    conn.close()

    return {
        "success":    True,
        "photo_id":   photo_id,
        "filename":   filename,
        "filepath":   filepath,
        "date":       date,
        "plant_id":   plant_id,
        "condition":  condition,
        "cv_prediction": cv_result,
        "message":    f"Photo uploaded successfully. CV prediction: {predicted_condition or 'N/A'}"
    }


@app.get("/photos/list")
def list_photos(condition: str = None, plant_id: str = None):
    """
    Returns all photos with optional filtering.
    Query params: ?condition=Healthy or ?plant_id=P01
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query  = "SELECT * FROM plant_photos WHERE 1=1"
    params = []

    if condition:
        query  += " AND (condition = ? OR predicted_condition = ?)"
        params += [condition, condition]
    if plant_id:
        query  += " AND plant_id = ?"
        params.append(plant_id)

    query += " ORDER BY uploaded_at DESC"
    cursor.execute(query, params)
    rows    = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    photos = [dict(zip(columns, row)) for row in rows]
    return {"total": len(photos), "photos": photos}


@app.get("/photos/{photo_id}/image")
def get_photo_image(photo_id: int):
    """Serve the actual image file for display in the dashboard gallery."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM plant_photos WHERE id = ?", (photo_id,))
    row    = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Photo not found")

    filepath = os.path.join(PHOTOS_DIR, row[0])
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Image file not found: {filepath}")

    return FileResponse(filepath)


@app.post("/photos/predict/{photo_id}")
def re_run_prediction(photo_id: int):
    """
    Re-runs the CV model on an already-uploaded photo.
    Useful after you retrain the model with more data.
    """
    if not CV_MODEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="CV model not available. Train it first.")

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM plant_photos WHERE id = ?", (photo_id,))
    row    = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Photo not found")

    filepath = os.path.join(PHOTOS_DIR, row[0])
    if not os.path.exists(filepath):
        conn.close()
        raise HTTPException(status_code=404, detail="Image file not found")

    result = predict_plant_condition(filepath)

    cursor.execute("""
        UPDATE plant_photos
        SET predicted_condition = ?, prediction_confidence = ?
        WHERE id = ?
    """, (result.get("predicted_class"), result.get("confidence"), photo_id))
    conn.commit()
    conn.close()

    return {
        "photo_id":   photo_id,
        "filename":   row[0],
        "cv_result":  result,
        "message":    "Prediction updated"
    }


@app.delete("/photos/{photo_id}")
def delete_photo(photo_id: int):
    """Delete a photo from both the folder and the database."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM plant_photos WHERE id = ?", (photo_id,))
    row    = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Photo not found")

    filepath = os.path.join(PHOTOS_DIR, row[0])
    if os.path.exists(filepath):
        os.remove(filepath)

    cursor.execute("DELETE FROM plant_photos WHERE id = ?", (photo_id,))
    conn.commit()
    conn.close()

    return {"success": True, "deleted_photo_id": photo_id, "deleted_file": row[0]}


@app.get("/health")
def health():
    return {
        "status":              "ok",
        "cv_model_available":  CV_MODEL_AVAILABLE,
        "photos_folder":       PHOTOS_DIR,
        "db_path":             DB_PATH
    }