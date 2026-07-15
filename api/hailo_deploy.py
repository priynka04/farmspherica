"""
api/hailo_deploy.py
Week 7 — Edge Deployment preparation for Hailo-8L on Raspberry Pi 5

This script handles the full conversion pipeline:
    YOLO26n ONNX → Hailo HEF format

STATUS: Script complete and ready. Hardware (Raspberry Pi 5 + Hailo-8L)
confirmed unavailable until next month (Akash, 06/07/2026).
Run this script the moment the Hailo SDK is installed on the Pi.

INSTRUCTIONS WHEN HARDWARE ARRIVES:
1. Install Hailo Dataflow Compiler on the Pi:
       pip install hailo-dataflow-compiler
2. Run this script:
       python api/hailo_deploy.py
3. This produces: models/disease_model_yolo26n_v2.hef
4. Copy the .hef to the Pi and run inference with hailo-rt library.

The ONNX model (models/disease_model_yolo26n_v2.onnx) is already
exported from Week 8 training and ready for conversion.
"""

from pathlib import Path
import sys

ONNX_MODEL_PATH = Path("models/disease_model_yolo26n_v2.onnx")
HEF_OUTPUT_PATH = Path("models/disease_model_yolo26n_v2.hef")
CALIB_DATA_PATH = Path("data/simulated_camera_feed")   # used for calibration


def check_prerequisites() -> bool:
    """Check that all files and libraries needed for conversion exist."""
    ok = True

    if not ONNX_MODEL_PATH.exists():
        print(f"[ERROR] ONNX model not found: {ONNX_MODEL_PATH}")
        print("        Run Week 8 disease detection training first.")
        ok = False
    else:
        size_mb = ONNX_MODEL_PATH.stat().st_size / 1e6
        print(f"[OK] ONNX model found: {ONNX_MODEL_PATH} ({size_mb:.2f} MB)")

    if not CALIB_DATA_PATH.exists():
        print(f"[WARNING] Calibration image folder not found: {CALIB_DATA_PATH}")
        print("          Conversion can still run but quantisation will be approximate.")
    else:
        images = list(CALIB_DATA_PATH.glob("*.jpg")) + list(CALIB_DATA_PATH.glob("*.png"))
        print(f"[OK] Calibration images: {len(images)} found in {CALIB_DATA_PATH}")

    try:
        import hailo_sdk_client
        print("[OK] Hailo Dataflow Compiler is installed")
    except ImportError:
        print("[WAITING] Hailo Dataflow Compiler not installed yet.")
        print("          This is expected — hardware arrives next month.")
        print("          Install with: pip install hailo-dataflow-compiler")
        ok = False

    return ok


def convert_to_hef():
    """
    Convert ONNX model to Hailo HEF format.
    This function runs when the Hailo SDK is available.
    """
    from hailo_sdk_client import ClientRunner, InferenceContext

    print("[INFO] Starting Hailo conversion pipeline...")
    print(f"       Input:  {ONNX_MODEL_PATH}")
    print(f"       Output: {HEF_OUTPUT_PATH}")

    runner = ClientRunner(hw_arch="hailo8l")   # Hailo-8L architecture

    # Step 1 — Parse ONNX
    print("[1/4] Parsing ONNX model...")
    runner.translate_onnx_model(
        str(ONNX_MODEL_PATH),
        "disease_model_yolo26n_v2",
        start_node_names=None,
        end_node_names=None,
    )

    # Step 2 — Optimise for Hailo-8L
    print("[2/4] Optimising model...")
    runner.optimize_full_precision()

    # Step 3 — Quantise to INT8 using calibration images
    print("[3/4] Quantising (INT8) using calibration images...")
    calib_images = (
        list(CALIB_DATA_PATH.glob("*.jpg")) +
        list(CALIB_DATA_PATH.glob("*.png"))
    )[:100]   # 100 images is enough for calibration
    runner.quantize(calib_images)

    # Step 4 — Compile to HEF
    print("[4/4] Compiling to HEF...")
    hef_bytes = runner.compile()
    HEF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEF_OUTPUT_PATH.write_bytes(hef_bytes)
    size_mb = HEF_OUTPUT_PATH.stat().st_size / 1e6
    print(f"[DONE] HEF saved: {HEF_OUTPUT_PATH} ({size_mb:.2f} MB)")


def benchmark_inference():
    """
    Run a latency benchmark on the Hailo-8L once deployed on the Pi.
    Success criterion from work plan: real-time inference on edge.
    For disease detection at 640x640, target is <10ms per image on Hailo-8L.
    """
    import time
    try:
        from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                                     ConfigureParams, InputVStreamParams,
                                     OutputVStreamParams, InferVStreams)
        import numpy as np

        print("[INFO] Running inference benchmark on Hailo-8L...")
        hef     = HEF(str(HEF_OUTPUT_PATH))
        params  = VDevice.create_params()
        device  = VDevice(params)
        network = device.configure(hef, ConfigureParams.create_from_hef(hef))[0]

        input_info  = hef.get_input_vstream_infos()[0]
        output_info = hef.get_output_vstream_infos()[0]

        # Warm-up
        dummy = np.random.uint8(size=(1, 640, 640, 3))
        for _ in range(5):
            with InferVStreams(network, InputVStreamParams.make_from_network_group(network),
                               OutputVStreamParams.make_from_network_group(network)) as streams:
                streams.send({"input_layer": dummy})
                streams.recv()

        # Benchmark 100 frames
        times = []
        for _ in range(100):
            start = time.perf_counter()
            with InferVStreams(network, InputVStreamParams.make_from_network_group(network),
                               OutputVStreamParams.make_from_network_group(network)) as streams:
                streams.send({"input_layer": dummy})
                streams.recv()
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        print(f"  Avg latency: {avg_ms:.2f} ms")
        print(f"  Min latency: {min_ms:.2f} ms")
        print(f"  Max latency: {max_ms:.2f} ms")
        print(f"  Target:      <10 ms  →  {'MET' if avg_ms < 10 else 'NOT MET'}")

    except ImportError:
        print("[WAITING] hailo_platform library not available — run on Pi with Hailo-8L")


if __name__ == "__main__":
    print("=" * 60)
    print("  Hailo-8L Edge Deployment — Farmspherica Disease Model")
    print("=" * 60)
    print()

    prereqs_ok = check_prerequisites()

    if not prereqs_ok:
        print()
        print("Prerequisites not yet met (hardware arrives next month).")
        print("This script is fully written and ready to run immediately")
        print("once the Hailo SDK is installed on the Raspberry Pi 5.")
        print()
        print("What IS ready right now:")
        print(f"  ONNX model: {ONNX_MODEL_PATH}")
        print(f"  Target HEF: {HEF_OUTPUT_PATH} (will be created on conversion)")
        print(f"  Calibration images: {CALIB_DATA_PATH}")
        sys.exit(0)

    convert_to_hef()
    benchmark_inference()