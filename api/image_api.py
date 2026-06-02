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
DB_PATH = "data/farmspherica.db"

os.makedirs(PHOTOS_DIR, exist_ok=True)

# Serve photos as static files
app.mount("/images", StaticFiles(directory=PHOTOS_DIR), name="photos")


def init_photos_table():
    """Creates the photos table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS plant_photos (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            filename      TEXT NOT NULL,
            date          TEXT NOT NULL,
            plant_id      TEXT NOT NULL,
            condition     TEXT NOT NULL,
            angle         TEXT,
            notes         TEXT,
            uploaded_at   TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# Run once at startup
init_photos_table()


# ──────────────────────────────────────────────────────────────
# Upload Photo
# ──────────────────────────────────────────────────────────────
@app.post("/photos/upload")
async def upload_photo(
    file: UploadFile = File(...),
    plant_id: str = Form(...),
    condition: str = Form(...),
    angle: str = Form("Front"),
    notes: str = Form(""),
    date: str = Form("")
):
    """
    Upload a plant photo.
    Image is stored in:
        photos/YYYY-MM-DD/
    Metadata is stored in SQLite.
    """

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    ext = os.path.splitext(file.filename)[1] or ".jpg"

    filename = f"{date}_{plant_id}_{condition}_{angle}{ext}"

    # Create date folder
    date_folder = os.path.join(PHOTOS_DIR, date)
    os.makedirs(date_folder, exist_ok=True)

    filepath = os.path.join(date_folder, filename)

    # Save image
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save metadata
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        INSERT INTO plant_photos
        (
            filename,
            date,
            plant_id,
            condition,
            angle,
            notes,
            uploaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        date,
        plant_id,
        condition,
        angle,
        notes,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

    return {
        "message": "Photo uploaded successfully",
        "filename": filename,
        "url": f"/images/{date}/{filename}"
    }


# ──────────────────────────────────────────────────────────────
# List Photos
# ──────────────────────────────────────────────────────────────
@app.get("/photos/list")
def list_photos(
    condition: str = None,
    plant_id: str = None
):
    """
    Returns all photos.
    Optional filters:
        ?condition=Healthy
        ?plant_id=P01
    """

    conn = sqlite3.connect(DB_PATH)

    query = "SELECT * FROM plant_photos WHERE 1=1"
    params = []

    if condition:
        query += " AND condition = ?"
        params.append(condition)

    if plant_id:
        query += " AND plant_id = ?"
        params.append(plant_id)

    query += " ORDER BY date DESC, uploaded_at DESC"

    rows = conn.execute(query, params).fetchall()

    conn.close()

    cols = [
        "id",
        "filename",
        "date",
        "plant_id",
        "condition",
        "angle",
        "notes",
        "uploaded_at"
    ]

    photos = []

    for row in rows:
        photo = dict(zip(cols, row))
        photo["url"] = f"/images/{photo['date']}/{photo['filename']}"
        photos.append(photo)

    return photos


# ──────────────────────────────────────────────────────────────
# Delete Photo
# ──────────────────────────────────────────────────────────────
@app.delete("/photos/{photo_id}")
def delete_photo(photo_id: int):
    """
    Deletes a photo from both
    the database and filesystem.
    """

    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        """
        SELECT filename, date
        FROM plant_photos
        WHERE id = ?
        """,
        (photo_id,)
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail="Photo not found"
        )

    filename, photo_date = row

    filepath = os.path.join(
        PHOTOS_DIR,
        photo_date,
        filename
    )

    if os.path.exists(filepath):
        os.remove(filepath)

    conn.execute(
        "DELETE FROM plant_photos WHERE id = ?",
        (photo_id,)
    )

    conn.commit()
    conn.close()

    return {
        "message": f"Deleted {filename}"
    }


# ──────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok"
    }