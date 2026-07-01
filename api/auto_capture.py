"""
api/auto_capture.py
Week 5 — Auto-Capture (SIMULATED, no ESP32-CAM hardware yet)

This script pretends to be the ESP32-CAM. Instead of fetching a photo from a
real camera URL, it picks an image from data/simulated_camera_feed/ on a
timer and saves it exactly the way a real capture would be saved.

WHEN THE REAL ESP32-CAM ARRIVES:
Only the capture_and_save() function's source needs to change — replace the
random.choice(...) + shutil.copy(...) lines with the real
requests.get(ESP32_CAM_URL) call shown in the commented-out REAL_CAMERA
version at the bottom of this file. Everything else (folder, filenames,
timer loop, downstream integration) stays exactly the same.

Run with:
    python api/auto_capture.py
"""

import random
import shutil
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIMULATED_FEED_DIR = Path("data/simulated_camera_feed")  # put sample photos here
SAVE_DIR = Path("data/auto_captures")                     # "captured" photos land here
CAPTURE_INTERVAL_SECONDS = 1800  # every 30 minutes — change for faster testing


def capture_and_save() -> Path:
    """Simulates one camera capture: picks a random sample image and saves
    it under a fresh timestamped filename, the same way a real capture would
    be named."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    sample_images = list(SIMULATED_FEED_DIR.glob("*.jpg")) + list(
        SIMULATED_FEED_DIR.glob("*.png")
    )
    if not sample_images:
        raise FileNotFoundError(
            f"No sample images found in {SIMULATED_FEED_DIR}. "
            "Add a few .jpg/.png lettuce photos there first."
        )

    chosen = random.choice(sample_images)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = SAVE_DIR / f"capture_{timestamp}{chosen.suffix}"
    shutil.copy(chosen, filepath)

    print(f"[SIMULATED] Saved: {filepath}  (source: {chosen.name})")
    return filepath


if __name__ == "__main__":
    print("Running in SIMULATED camera mode — no real ESP32-CAM connected yet.")
    print(f"Pulling sample images from: {SIMULATED_FEED_DIR}")
    print(f"Capture interval: every {CAPTURE_INTERVAL_SECONDS} seconds\n")

    while True:
        try:
            capture_and_save()
        except Exception as e:
            print(f"Capture failed: {e}")
        time.sleep(CAPTURE_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# REAL CAMERA VERSION (swap in once the ESP32-CAM module is available)
# Replace the capture_and_save() function above with this version — nothing
# else in this file or in any code that calls capture_and_save() needs to
# change, since it still returns a saved file path either way.
# ---------------------------------------------------------------------------
#
# import requests
#
# ESP32_CAM_URL = "http://<ESP32_IP_ADDRESS>/capture"
#
# def capture_and_save() -> Path:
#     SAVE_DIR.mkdir(parents=True, exist_ok=True)
#     response = requests.get(ESP32_CAM_URL, timeout=10)
#     response.raise_for_status()
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filepath = SAVE_DIR / f"capture_{timestamp}.jpg"
#     with open(filepath, "wb") as f:
#         f.write(response.content)
#     print(f"[REAL CAMERA] Saved: {filepath}")
#     return filepath