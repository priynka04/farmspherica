"""
tests/test_segmentation.py
Automated tests for Week 5 segmentation pipeline.

Run with:
    python tests/test_segmentation.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))


def test_model_file_exists():
    assert Path("models/segmentation_model_v1.pt").exists(), \
        "segmentation_model_v1.pt missing — copy it to models/ first"
    print("Test 1 passed: model file exists")


def test_auto_capture_folder_exists():
    assert Path("data/simulated_camera_feed").exists(), \
        "data/simulated_camera_feed folder missing"
    images = list(Path("data/simulated_camera_feed").glob("*.jpg"))
    assert len(images) > 0, \
        "No .jpg images in data/simulated_camera_feed — add some first"
    print(f"Test 2 passed: simulated camera feed has {len(images)} images")


def test_segmentation_runs_on_sample():
    from api.segmentation import analyse_image
    images = list(Path("data/simulated_camera_feed").glob("*.jpg"))
    result = analyse_image(images[0])
    assert "leaf_count" in result
    assert "canopy_area_px" in result
    assert "canopy_area_cm2" in result
    assert "biomass_g" in result
    assert isinstance(result["leaf_count"], int)
    assert result["canopy_area_px"] >= 0
    print(f"Test 3 passed: segmentation ran — {result['leaf_count']} leaf/leaves detected")


def test_output_image_saved():
    from api.segmentation import analyse_image
    images = list(Path("data/simulated_camera_feed").glob("*.jpg"))
    result = analyse_image(images[0])
    assert Path(result["output_image"]).exists(), \
        f"Annotated output image not saved at {result['output_image']}"
    print(f"Test 4 passed: annotated image saved to {result['output_image']}")


def test_auto_capture_runs():
    from api.auto_capture import capture_and_save
    saved_path = capture_and_save()
    assert saved_path.exists(), f"Captured file not found at {saved_path}"
    print(f"Test 5 passed: auto_capture saved a file at {saved_path}")


if __name__ == "__main__":
    test_model_file_exists()
    test_auto_capture_folder_exists()
    test_segmentation_runs_on_sample()
    test_output_image_saved()
    test_auto_capture_runs()
    print("\nAll Week 5 segmentation tests passed!")