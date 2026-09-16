"""
Phase 4 — Diagnose a single prediction from the trained model.

Run this against an image that the trained model is misclassifying,
to see the EXACT numbers driving the decision — raw forensic values,
log-odds, standardized (z-score) values, and per-class probabilities.

This is the fastest way to find out WHY something is misclassified:
if a z-score for one feature is huge (e.g. +8 or -8), that feature's
value for this image is way outside the range the model saw during
training — a strong sign of dataset distribution mismatch between
your training data (RAISE/CASIA/CocoGlide) and real-world upload
photos.

Usage:
    python diagnose_prediction.py path/to/image.jpg
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from cv.image_reader import read_image
from cv.ela import generate_difference_image, calculate_ela_metrics
from cv.forgery import analyze_forgery
from cv.manipulation import analyze_manipulation
from ai.detector import detect_ai_image

from services.final_fusion import (
    _FORENSIC_FEATURE_NAMES,
    _probability_to_log_odds,
    _softmax,
)

from phase4_common import FINAL_MODEL_FILE


def main():

    if len(sys.argv) < 2:
        print("Usage: python diagnose_prediction.py path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    if not FINAL_MODEL_FILE.exists():
        print(f"No trained model found at {FINAL_MODEL_FILE}")
        print("(re-enable it first if you renamed it to .disabled)")
        sys.exit(1)

    with FINAL_MODEL_FILE.open(encoding="utf-8") as file:
        model = json.load(file)

    # --------------------------------------------------------
    # Run the exact same pipeline as app.py
    # --------------------------------------------------------

    image_bytes = image_path.read_bytes()

    success, image_cv, _info, error = read_image(image_bytes)

    if not success:
        print(f"Could not read image: {error}")
        sys.exit(1)

    suffix = image_path.suffix or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(image_bytes)
        temp_path = temp_file.name

    try:
        ai_detection = detect_ai_image(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    difference_image = generate_difference_image(image_bytes)
    ela_metrics = calculate_ela_metrics(difference_image)
    forgery_analysis = analyze_forgery(image_cv)
    manipulation_analysis = analyze_manipulation(image_cv, difference_image)

    forensic = {
        "meanDifference": ela_metrics["meanDifference"],
        "maxDifference": ela_metrics["maxDifference"],
        "standardDeviation": ela_metrics["standardDeviation"],
        "edgeDensity": forgery_analysis["edgeDensity"],
        "noiseMean": forgery_analysis["noiseMean"],
        "noiseStd": forgery_analysis["noiseStd"],
        "blurScore": forgery_analysis["blurScore"],
        "sharpnessScore": forgery_analysis["sharpnessScore"],
        "blockSharpnessAverage": forgery_analysis["blockSharpnessAverage"],
        "blockSharpnessStd": forgery_analysis["blockSharpnessStd"],
    }

    ai_probability = ai_detection["ai_probability"]
    manipulation_probability = manipulation_analysis["manipulationProbability"]

    # --------------------------------------------------------
    # Build the exact feature vector the model sees
    # --------------------------------------------------------

    feature_names = ["ai_log_odds", "manipulation_log_odds", *_FORENSIC_FEATURE_NAMES]

    feature_values = [
        _probability_to_log_odds(ai_probability),
        _probability_to_log_odds(manipulation_probability),
        *[forensic[name] for name in _FORENSIC_FEATURE_NAMES],
    ]

    features = np.array(feature_values, dtype=np.float64)
    mean = np.array(model["mean"], dtype=np.float64)
    std = np.array(model["std"], dtype=np.float64)
    weights = np.array(model["weights"], dtype=np.float64)
    bias = np.array(model["bias"], dtype=np.float64)
    class_names = model["class_names"]

    z_scores = (features - mean) / std

    logits = z_scores @ weights + bias
    probabilities = _softmax(logits)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("=" * 78)
    print(f"DIAGNOSTIC: {image_path.name}")
    print("=" * 78)

    print(f"\nRaw signals:")
    print(f"  ai_probability            = {ai_probability}")
    print(f"  manipulationProbability   = {manipulation_probability}")

    print(f"\n{'Feature':<24s}{'raw value':>14s}{'train mean':>14s}{'train std':>12s}{'z-score':>10s}")

    for name, value, m, s, z in zip(feature_names, feature_values, mean, std, z_scores):

        flag = "  <-- FAR OUTSIDE TRAINING RANGE" if abs(z) > 3 else ""

        print(f"{name:<24s}{value:>14.4f}{m:>14.4f}{s:>12.4f}{z:>10.2f}{flag}")

    print(f"\nPredicted class probabilities:")

    for name, prob in sorted(zip(class_names, probabilities.tolist()), key=lambda p: -p[1]):
        print(f"  {name:<22s} {prob * 100:.2f}%")

    print("\n" + "=" * 78)
    print(
        "If several features show '<-- FAR OUTSIDE TRAINING RANGE' "
        "(|z-score| > 3), this image's forensic profile doesn't look "
        "like anything the model saw during training — a sign the "
        "training dataset (RAISE/CASIA/CocoGlide) doesn't represent "
        "real-world upload photos well enough yet. Share this output "
        "so we can pin down which specific feature(s) are the problem."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()