"""
Phase 4 — Feature extraction over the datasets/ folder.

Run this on YOUR machine (not in a sandbox) — it needs the real
Community Forensics ViT model (ai/detector.py's MODEL_DIR) to compute
ai_probability for every image, same as the live app.py pipeline.

Usage:
    python extract_features.py

Expects:
    python-engine/datasets/authentic/...
    python-engine/datasets/manipulated/...
    python-engine/datasets/partially_ai_edited/...
    python-engine/datasets/ai_generated/...

(see phase4_common.py docstring for full folder-structure details)

Produces:
    python-engine/phase4_dataset_features.csv

Resumable: if the output CSV already has a row for a given filename,
that image is skipped — so you can re-run this after it's
interrupted partway through a large dataset without starting over.
"""

import csv
import random
import sys
import traceback
from pathlib import Path

from cv.image_reader import read_image
from cv.ela import generate_difference_image, calculate_ela_metrics
from cv.forgery import analyze_forgery
from cv.manipulation import analyze_manipulation
from ai.detector import detect_ai_image

from phase4_common import (
    DATASET_DIR,
    CLASS_FOLDERS,
    FEATURES_FILE,
    CSV_COLUMNS,
    FORENSIC_FEATURES,
    MAX_IMAGES_PER_CLASS,
    SAMPLING_SEED,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_already_processed():
    if not FEATURES_FILE.exists():
        return set()

    with FEATURES_FILE.open(encoding="utf-8") as file:
        return {
            row["filename"]
            for row in csv.DictReader(file)
        }


def process_image(image_path, class_label):
    """
    Run the same analysis pipeline app.py uses on a live upload,
    against one dataset image, and return a CSV-ready row dict.
    """

    image_bytes = image_path.read_bytes()

    success, image_cv, _info, error = read_image(image_bytes)

    if not success:
        raise ValueError(f"Could not read image: {error}")

    temp_path = None

    try:

        import tempfile

        suffix = image_path.suffix or ".jpg"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:

            temp_file.write(image_bytes)
            temp_path = temp_file.name

        ai_detection = detect_ai_image(temp_path)

    finally:

        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    difference_image = generate_difference_image(image_bytes)

    ela_metrics = calculate_ela_metrics(difference_image)

    forgery_analysis = analyze_forgery(image_cv)

    manipulation_analysis = analyze_manipulation(
        image_cv,
        difference_image
    )

    row = {
        "filename": str(image_path.relative_to(DATASET_DIR)),
        "class_label": class_label,
        "group_id": "",  # filled in later by split_dataset()
        "ai_probability": ai_detection["ai_probability"],
        "manipulationProbability": manipulation_analysis["manipulationProbability"],
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

    return row


def main():

    if not DATASET_DIR.exists():
        print(f"ERROR: {DATASET_DIR} does not exist.")
        print("Create it and add class subfolders — see phase4_common.py docstring.")
        sys.exit(1)

    already_processed = load_already_processed()

    write_header = not FEATURES_FILE.exists()

    total_written = 0
    total_skipped = 0
    total_failed = 0

    with FEATURES_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)

        if write_header:
            writer.writeheader()

        for folder_name, class_label in CLASS_FOLDERS.items():

            class_folder = DATASET_DIR / folder_name

            if not class_folder.exists():
                print(f"Skipping missing folder: {class_folder}")
                continue

            image_paths = find_images(class_folder)

            cap = MAX_IMAGES_PER_CLASS.get(class_label)

            if cap is not None and len(image_paths) > cap:

                rng = random.Random(SAMPLING_SEED)

                image_paths = rng.sample(image_paths, cap)

                image_paths.sort()

                print(
                    f"\n{class_label}: {len(find_images(class_folder))} found, "
                    f"randomly sampled down to {cap} (seed={SAMPLING_SEED})"
                )

            else:

                print(f"\n{class_label}: {len(image_paths)} image(s) found in {class_folder}")

            for index, image_path in enumerate(image_paths, start=1):

                relative_name = str(image_path.relative_to(DATASET_DIR))

                if relative_name in already_processed:
                    total_skipped += 1
                    continue

                try:

                    row = process_image(image_path, class_label)

                    writer.writerow(row)
                    file.flush()

                    total_written += 1

                except Exception as error:

                    total_failed += 1

                    print(f"  FAILED: {relative_name} — {error}")
                    traceback.print_exc(limit=1)

                if index % 25 == 0:
                    print(f"  ...{index}/{len(image_paths)} processed")

    print("\n" + "=" * 60)
    print(f"New rows written : {total_written}")
    print(f"Already done     : {total_skipped}")
    print(f"Failed           : {total_failed}")
    print(f"Output file      : {FEATURES_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()