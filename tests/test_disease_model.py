# =============================================================
# FILE: tests/test_disease_model.py
# WHAT IT TESTS:
#   1. Model file exists
#   2. Model loads successfully
#   3. Prediction returns correct structure
#   4. Prediction returns a valid class name
#   5. All 5 disease classes are recognized
#   6. Confidence is between 0 and 1
#   7. All export formats exist
#   8. Missing file handled gracefully
# HOW TO RUN: python tests/test_disease_model.py
# =============================================================

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw
import random


# ── Helpers ──────────────────────────────────────────────────

def create_test_image(path, color=(34, 139, 34)):
    """Creates a simple green test image (simulates a plant)."""
    img = Image.new('RGB', (640, 640), color=color)
    draw = ImageDraw.Draw(img)
    for _ in range(5):
        x = random.randint(50, 500)
        y = random.randint(50, 500)
        size = random.randint(30, 100)
        draw.ellipse([(x, y), (x + size, y + size)],
                     fill=(random.randint(0, 80),
                           random.randint(100, 200),
                           random.randint(0, 80)))
    img.save(path)


def print_result(test_name, passed, detail=""):
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{status}  {test_name}")
    if detail:
        print(f"         → {detail}")


# ── Tests ────────────────────────────────────────────────────

def test_1_model_file_exists():
    """Test 1: YOLO26n .pt model file exists in models/"""
    path = "models/disease_model_yolo26n_v2.pt"
    passed = os.path.exists(path)
    size_mb = os.path.getsize(path) / (1024 * 1024) if passed else 0
    print_result(
        "Model file exists",
        passed,
        f"{path} ({size_mb:.2f} MB)" if passed else f"{path} NOT FOUND"
    )
    return passed


def test_2_model_loads():
    """Test 2: Model loads without errors."""
    try:
        from api.cv_model import load_model
        model = load_model()
        passed = model is not None
        print_result("Model loads successfully", passed,
                     "YOLO26n loaded and cached" if passed else "load_model() returned None")
        return passed
    except Exception as e:
        print_result("Model loads successfully", False, str(e))
        return False


def test_3_prediction_structure():
    """Test 3: predict_plant_condition returns all required keys."""
    try:
        from api.cv_model import predict_plant_condition

        os.makedirs("photos", exist_ok=True)
        img_path = "photos/_test_structure.jpg"
        create_test_image(img_path)

        result = predict_plant_condition(img_path)

        required_keys = [
            "predicted_class",
            "confidence",
            "confidence_pct",
            "class_id",
            "model_version",
            "image_path",
            "detection_count"
        ]
        missing = [k for k in required_keys if k not in result]
        passed = len(missing) == 0

        print_result(
            "Prediction returns correct structure",
            passed,
            f"All {len(required_keys)} keys present" if passed
            else f"Missing keys: {missing}"
        )

        if os.path.exists(img_path):
            os.remove(img_path)

        return passed
    except Exception as e:
        print_result("Prediction returns correct structure", False, str(e))
        return False


def test_4_prediction_returns_valid_class():
    """
    Test 4: Prediction always returns a valid class name.
    Note: synthetic images may trigger disease detection (that is correct
    model behaviour — the model detects visual patterns, not real plants).
    What matters is the returned class is one of the 6 valid values.
    """
    try:
        from api.cv_model import predict_plant_condition, DISEASE_CLASSES

        os.makedirs("photos", exist_ok=True)
        img_path = "photos/_test_valid_class.jpg"
        create_test_image(img_path, color=(34, 139, 34))

        result = predict_plant_condition(img_path)
        predicted = result.get("predicted_class")

        # Valid classes = 5 diseases + Healthy
        valid_classes = list(DISEASE_CLASSES.values()) + ["Healthy"]
        passed = predicted in valid_classes

        print_result(
            "Prediction returns a valid class name",
            passed,
            f"predicted_class = '{predicted}' ✅ (valid)" if passed
            else f"predicted_class = '{predicted}' ❌ (not in valid list)"
        )

        if os.path.exists(img_path):
            os.remove(img_path)

        return passed
    except Exception as e:
        print_result("Prediction returns a valid class name", False, str(e))
        return False


def test_5_all_classes_defined():
    """Test 5: All 5 disease classes are defined in cv_model.py."""
    try:
        from api.cv_model import DISEASE_CLASSES

        expected = {
            0: "Bacterial",
            1: "Downy_mildew_on_lettuce",
            2: "Lettuce Mosaic Virus",
            3: "Powdery_mildew_on_lettuce",
            4: "Septoria_Blight_on_lettuce"
        }

        passed = DISEASE_CLASSES == expected
        print_result(
            "All 5 disease classes defined correctly",
            passed,
            f"Classes: {list(DISEASE_CLASSES.values())}" if passed
            else f"Mismatch. Got: {DISEASE_CLASSES}"
        )
        return passed
    except Exception as e:
        print_result("All 5 disease classes defined correctly", False, str(e))
        return False


def test_6_confidence_range():
    """Test 6: Confidence score is between 0.0 and 1.0."""
    try:
        from api.cv_model import predict_plant_condition

        os.makedirs("photos", exist_ok=True)
        img_path = "photos/_test_conf.jpg"
        create_test_image(img_path)

        result = predict_plant_condition(img_path)
        confidence = result.get("confidence", -1)
        passed = 0.0 <= confidence <= 1.0

        print_result(
            "Confidence score is between 0.0 and 1.0",
            passed,
            f"confidence = {confidence:.4f}"
        )

        if os.path.exists(img_path):
            os.remove(img_path)

        return passed
    except Exception as e:
        print_result("Confidence score is between 0.0 and 1.0", False, str(e))
        return False


def test_7_all_exports_exist():
    """Test 7: All 4 export format files exist in models/."""
    exports = {
        "PyTorch (.pt)": "models/disease_model_yolo26n_v2.pt",
        "ONNX (.onnx)":  "models/disease_model_yolo26n_v2.onnx",
        "TFLite fp16":   "models/disease_model_yolo26n_v2_float16.tflite",
        "TFLite fp32":   "models/disease_model_yolo26n_v2_float32.tflite",
    }

    all_passed = True
    for name, path in exports.items():
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"         ✅  {name}: {size:.2f} MB")
        else:
            print(f"         ❌  {name}: MISSING — {path}")
            all_passed = False

    print_result(
        "All 4 export formats exist",
        all_passed,
        "PyTorch, ONNX, TFLite-fp16, TFLite-fp32" if all_passed
        else "Some files missing"
    )
    return all_passed


def test_8_missing_file_handled():
    """Test 8: predict_plant_condition handles missing image gracefully."""
    try:
        from api.cv_model import predict_plant_condition

        result = predict_plant_condition("photos/this_does_not_exist.jpg")
        passed = "error" in result or result.get("predicted_class") is None

        print_result(
            "Missing image handled gracefully (no crash)",
            passed,
            f"returned error dict correctly"
        )
        return passed
    except Exception as e:
        print_result("Missing image handled gracefully", False,
                     f"Raised exception instead: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DISEASE MODEL TEST SUITE — YOLO26n")
    print("  Model: disease_model_yolo26n_v2.pt")
    print("=" * 60)
    print()

    results = {
        "Test 1 — Model file exists":              test_1_model_file_exists(),
        "Test 2 — Model loads":                    test_2_model_loads(),
        "Test 3 — Prediction structure":           test_3_prediction_structure(),
        "Test 4 — Valid class returned":           test_4_prediction_returns_valid_class(),
        "Test 5 — All 5 classes defined":          test_5_all_classes_defined(),
        "Test 6 — Confidence range":               test_6_confidence_range(),
        "Test 7 — All exports exist":              test_7_all_exports_exist(),
        "Test 8 — Missing file handled":           test_8_missing_file_handled(),
    }

    print()
    print("=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"  RESULT: {passed}/{total} tests passed")

    if passed == total:
        print("  ✅ ALL TESTS PASSED — YOLO26n ready for production!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  ❌ {total - passed} test(s) failed:")
        for f in failed:
            print(f"     - {f}")
    print("=" * 60)