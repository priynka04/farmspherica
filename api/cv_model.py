# =============================================================
# FILE: api/cv_model.py  (Week 4 — Disease Detection)
# WHAT IT DOES:
#   - Loads the trained YOLOv8 disease detection model
#   - Runs inference on plant photos
#   - Returns predicted disease class + confidence
# =============================================================

import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Path to the trained model
MODEL_PATH = "models/disease_model_yolo26n_v2.pt"

# Disease class names (must match training labels)
DISEASE_CLASSES = {
    0: "Bacterial",
    1: "Downy_mildew_on_lettuce",
    2: "Lettuce Mosaic Virus",
    3: "Powdery_mildew_on_lettuce",
    4: "Septoria_Blight_on_lettuce"
}

# Global model cache (load once, reuse)
_model = None

def load_model():
    """Load the trained YOLO model once and cache it."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                f"Download disease_model_v2.pt from Colab and save it here."
            )
        print(f"[MODEL] Loading {MODEL_PATH}...")
        _model = YOLO(MODEL_PATH)
        print("[MODEL] Loaded successfully ✅")
    return _model


def predict_plant_condition(filepath):
    """
    Run disease detection on a plant photo.
    
    Args:
        filepath (str): Path to the image file
        
    Returns:
        dict: {
            "predicted_class": "Disease_Name",
            "confidence": 0.95,
            "confidence_pct": "95%",
            "class_id": 0,
            "model_version": "v2",
            "image_path": filepath
        }
    """
    if not os.path.exists(filepath):
        return {
            "error": f"Image file not found: {filepath}",
            "predicted_class": None,
            "confidence": 0.0
        }
    
    try:
        model = load_model()
        
        # Run inference
        # conf=0.25 is the confidence threshold (only return detections with conf > 0.25)
        results = model.predict(
            source=filepath,
            conf=0.25,
            verbose=False,
            imgsz=640
        )
        
        # Extract the best prediction from results
        if results and len(results) > 0:
            result = results[0]
            
            # If there are detections
            if result.boxes is not None and len(result.boxes) > 0:
                # Get the box with highest confidence
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                best_idx = np.argmax(confidences)
                best_confidence = float(confidences[best_idx])
                best_class_id = int(class_ids[best_idx])
                
                predicted_class = DISEASE_CLASSES.get(best_class_id, "Unknown")
                
                return {
                    "predicted_class": predicted_class,
                    "confidence": round(best_confidence, 4),
                    "confidence_pct": f"{int(best_confidence * 100)}%",
                    "class_id": best_class_id,
                    "model_version": "v2",
                    "image_path": filepath,
                    "detection_count": len(result.boxes)
                }
            else:
                # No disease detected (healthy plant)
                return {
                    "predicted_class": "Healthy",
                    "confidence": 1.0,
                    "confidence_pct": "100%",
                    "class_id": -1,
                    "model_version": "v2",
                    "image_path": filepath,
                    "detection_count": 0
                }
        else:
            return {
                "error": "No results from model",
                "predicted_class": None,
                "confidence": 0.0
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "predicted_class": None,
            "confidence": 0.0,
            "image_path": filepath
        }


if __name__ == "__main__":
    # Quick test
    test_image = "photos/test.jpg"
    if os.path.exists(test_image):
        result = predict_plant_condition(test_image)
        print(f"Test result: {result}")
    else:
        print(f"No test image found at {test_image}")