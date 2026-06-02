from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
import shutil
from datetime import datetime

app = FastAPI(title="Farmspherica Image Logger")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PHOTOS_DIR = "photos"
DB_PATH    = "data/farmspherica.db"
os.makedirs(PHOTOS_DIR, exist_ok=True)

# Serve the photos/ folder as static files
# so the frontend can display images by URL
app.mount("/images", StaticFiles(directory=PHOTOS_DIR), name="photos")

def init_photos_table():
    """Creates the photos table in the database if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plant_photos (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            filename  TEXT    NOT NULL,
            date      TEXT    NOT NULL,
            plant_id  TEXT    NOT NULL,
            condition TEXT    NOT NULL,
            angle     TEXT,
            notes     TEXT,
            uploaded_at TEXT  NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_photos_table()  # runs once when the API starts

# ── Endpoint 1: upload a photo ───────────────────────────────────────────────
@app.post("/photos/upload")
async def upload_photo(
    file:      UploadFile = File(...),
    plant_id:  str        = Form(...),
    condition: str        = Form(...),
    angle:     str        = Form("Front"),
    notes:     str        = Form(""),
    date:      str        = Form("")
):
    """
    Upload a plant photo.
    The photo is saved to photos/ folder.
    Metadata is saved to the database.
    """
    if date == "":
        date = datetime.now().strftime("%Y-%m-%d")

    # Build filename: 2026-05-20_P01_Healthy_Front.jpg
    ext      = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"{date}_{plant_id}_{condition}_{angle}{ext}"
    filepath = os.path.join(PHOTOS_DIR, filename)

    # Save the actual image file to the photos/ folder
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save metadata to the database
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO plant_photos
            (filename, date, plant_id, condition, angle, notes, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (filename, date, plant_id, condition, angle, notes,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()

    return {
        "message":  "Photo uploaded successfully",
        "filename": filename,
        "url":      f"/images/{filename}"
    }

# ── Endpoint 2: list all photos ──────────────────────────────────────────────
@app.get("/photos/list")
def list_photos(condition: str = None, plant_id: str = None):
    """
    Returns all photos from the database.
    Optional filters: condition=Healthy, plant_id=P01
    """
    conn   = sqlite3.connect(DB_PATH)
    query  = "SELECT * FROM plant_photos WHERE 1=1"
    params = []
    if condition:
        query  += " AND condition = ?"
        params.append(condition)
    if plant_id:
        query  += " AND plant_id = ?"
        params.append(plant_id)
    query += " ORDER BY date DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    cols   = ["id","filename","date","plant_id","condition","angle","notes","uploaded_at"]
    return [dict(zip(cols, r)) for r in rows]

# ── Endpoint 3: delete a photo ───────────────────────────────────────────────
@app.delete("/photos/{photo_id}")
def delete_photo(photo_id: int):
    """Deletes a photo from both the database and the folder."""
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        "SELECT filename FROM plant_photos WHERE id = ?", (photo_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Photo not found")
    filename = row[0]
    filepath = os.path.join(PHOTOS_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    conn.execute("DELETE FROM plant_photos WHERE id = ?", (photo_id,))
    conn.commit()
    conn.close()
    return {"message": f"Deleted {filename}"}

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}