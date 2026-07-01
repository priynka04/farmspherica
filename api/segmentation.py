"""
api/segmentation.py
Week 5 — Leaf Segmentation: auto leaf count + canopy area + biomass estimate

Given any plant photo (from auto_capture.py or a manual upload), this script:
  1. Loads the trained YOLO26 segmentation model
  2. Detects and outlines every leaf in the photo
  3. Counts total leaves
  4. Measures total canopy area in pixels
  5. Estimates biomass using a published lettuce calibration factor
  6. Saves the result to the database (plant_photos table)
  7. Saves the annotated image (with blue leaf outlines) to disk

Run directly to test on a single image:
    python api/segmentation.py

Or import and call process_captured_image(image_path) from other scripts.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/segmentation_model_v1.pt")
DB_PATH = Path("data/farmspherica.db")
OUTPUT_DIR = Path("data/segmentation_outputs")

# Biomass calibration factor for lettuce (fresh weight estimate).
# Source: Bauer et al. (2019) — fresh weight (g) ≈ canopy_area_cm2 × 0.37
# Camera at ~50cm height: 640x640 image covers ~30cm×30cm
# → each pixel ≈ 0.0022 cm²  (update once real camera height is known)
PIXEL_TO_CM2 = 0.0022
BIOMASS_FACTOR = 0.37
CONF_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Load model once and cache it
# ---------------------------------------------------------------------------
_model = None


def load_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Segmentation model not found at {MODEL_PATH}. "
                "Make sure models/segmentation_model_v1.pt is in place."
            )
        _model = YOLO(str(MODEL_PATH))
        print(f"[INFO] Segmentation model loaded from {MODEL_PATH}")
    return _model


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------
def analyse_image(image_path) -> dict:
    """
    Run leaf segmentation on one image.
    Returns a dict with leaf_count, canopy_area_px, canopy_area_cm2,
    biomass_g, confidence_avg, output_image, timestamp, source_image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        str(image_path), conf=CONF_THRESHOLD, save=False, verbose=False
    )

    r = results[0]

    leaf_count = 0
    canopy_area_px = 0
    confidences = []

    if r.masks is not None and len(r.masks.data) > 0:
        leaf_count = len(r.masks.data)
        for mask in r.masks.data:
            canopy_area_px += int(mask.sum().item())
        if r.boxes is not None:
            confidences = r.boxes.conf.tolist()

    canopy_area_cm2 = round(canopy_area_px * PIXEL_TO_CM2, 2)
    biomass_g = round(canopy_area_cm2 * BIOMASS_FACTOR, 2)
    confidence_avg = (
        round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    annotated = r.plot()
    out_filename = (
        f"seg_{image_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    )
    out_path = OUTPUT_DIR / out_filename
    cv2.imwrite(str(out_path), annotated)

    result = {
        "leaf_count": leaf_count,
        "canopy_area_px": canopy_area_px,
        "canopy_area_cm2": canopy_area_cm2,
        "biomass_g": biomass_g,
        "confidence_avg": confidence_avg,
        "output_image": str(out_path),
        "timestamp": timestamp,
        "source_image": str(image_path),
    }

    print(f"[SEGMENTATION] {image_path.name}")
    print(f"  Leaves detected:  {leaf_count}")
    print(f"  Canopy area:      {canopy_area_px} px  ({canopy_area_cm2} cm2)")
    print(f"  Biomass estimate: {biomass_g} g")
    print(f"  Avg confidence:   {confidence_avg}")
    print(f"  Annotated image:  {out_path}")

    return result


# ---------------------------------------------------------------------------
# Save result to database
# ---------------------------------------------------------------------------
def save_to_db(result: dict, plant_id: str = "AUTO") -> int:
    """
    Save one segmentation result into the plant_photos table.
    Handles all NOT NULL columns: filename, date, plant_id, condition, uploaded_at.
    """
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Add any missing segmentation columns (safe to run multiple times)
    existing_cols = [row[1] for row in cur.execute("PRAGMA table_info(plant_photos)")]
    new_cols = {
        "auto_leaf_count": "INTEGER",
        "canopy_area_px": "INTEGER",
        "canopy_area_cm2": "REAL",
        "biomass_estimate_g": "REAL",
        "seg_confidence": "REAL",
        "seg_output_image": "TEXT",
        "capture_timestamp": "TEXT",
    }
    for col, col_type in new_cols.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE plant_photos ADD COLUMN {col} {col_type}")

    now = result["timestamp"]

    cur.execute(
        """
        INSERT INTO plant_photos
            (filename, date, plant_id, condition, uploaded_at,
             capture_timestamp,
             auto_leaf_count, canopy_area_px, canopy_area_cm2,
             biomass_estimate_g, seg_confidence, seg_output_image)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            result["source_image"],  # filename   (NOT NULL)
            now[:10],  # date        (NOT NULL) e.g. "2026-07-01"
            plant_id,  # plant_id    (NOT NULL)
            "auto_capture",  # condition   (NOT NULL) — auto-captured, not manually labelled
            now,  # uploaded_at (NOT NULL)
            now,  # capture_timestamp
            result["leaf_count"],
            result["canopy_area_px"],
            result["canopy_area_cm2"],
            result["biomass_g"],
            result["confidence_avg"],
            result["output_image"],
        ),
    )

    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    print(f"[DB] Saved — row id: {row_id}")
    return row_id


# ---------------------------------------------------------------------------
# One-call pipeline used by auto_capture.py
# ---------------------------------------------------------------------------
def process_captured_image(image_path, plant_id: str = "AUTO") -> dict:
    """
    Analyse image + save to DB in one step.
    Call this from auto_capture.py after every new photo is saved.
    """
    result = analyse_image(image_path)
    result["db_row_id"] = save_to_db(result, plant_id=plant_id)
    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Segmentation Self-Test ===\n")

    test_dir = Path("data/simulated_camera_feed")
    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

    if not images:
        print(f"No images found in {test_dir}.")
        print("Add some .jpg images there first.")
    else:
        test_img = images[0]
        print(f"Running on: {test_img}\n")
        result = process_captured_image(test_img, plant_id="TEST_PLANT")

        print("\n=== Result Summary ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("\nSelf-test complete!")
