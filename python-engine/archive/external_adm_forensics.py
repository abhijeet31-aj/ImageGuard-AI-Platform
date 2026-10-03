import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

REAL_DIR = BASE_DIR / "external-test" / "ADM" / "REAL"
AI_DIR = BASE_DIR / "external-test" / "ADM" / "AI"

OUTPUT_FILE = BASE_DIR / "external_adm_forensic_results.csv"


# ==========================================
# ELA
# ==========================================

def calculate_ela(image_path):

    original = Image.open(image_path).convert("RGB")

    temp_path = BASE_DIR / "_ela_temp.jpg"

    original.save(
        temp_path,
        "JPEG",
        quality=90
    )

    recompressed = Image.open(temp_path).convert("RGB")

    difference = ImageChops.difference(
        original,
        recompressed
    )

    diff_array = np.array(difference).astype(np.float32)

    mean_difference = float(diff_array.mean())
    max_difference = float(diff_array.max())
    standard_deviation = float(diff_array.std())

    if temp_path.exists():
        temp_path.unlink()

    return (
        mean_difference,
        max_difference,
        standard_deviation
    )


# ==========================================
# FORENSIC FEATURES
# ==========================================

def calculate_forensics(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------
    # Edge Density
    # --------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = float(
        np.mean(edges > 0) * 100
    )

    # --------------------------------------
    # Noise
    # --------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    noise = gray.astype(
        np.float32
    ) - blurred.astype(
        np.float32
    )

    noise_mean = float(
        np.mean(np.abs(noise))
    )

    noise_std = float(
        np.std(noise)
    )

    # --------------------------------------
    # Blur Score
    # --------------------------------------

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    # --------------------------------------
    # Sharpness Score
    # --------------------------------------

    sharpness_score = float(
        np.mean(
            np.abs(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F
                )
            )
        )
    )

    # --------------------------------------
    # Block Sharpness
    # --------------------------------------

    h, w = gray.shape

    block_size = 64

    block_scores = []

    for y in range(
        0,
        h - block_size + 1,
        block_size
    ):

        for x in range(
            0,
            w - block_size + 1,
            block_size
        ):

            block = gray[
                y:y + block_size,
                x:x + block_size
            ]

            score = float(
                cv2.Laplacian(
                    block,
                    cv2.CV_64F
                ).var()
            )

            block_scores.append(score)

    if block_scores:

        block_sharpness_average = float(
            np.mean(block_scores)
        )

        block_sharpness_std = float(
            np.std(block_scores)
        )

    else:

        block_sharpness_average = 0.0
        block_sharpness_std = 0.0


    return {
        "edgeDensity": edge_density,
        "noiseMean": noise_mean,
        "noiseStd": noise_std,
        "blurScore": blur_score,
        "sharpnessScore": sharpness_score,
        "blockSharpnessAverage": block_sharpness_average,
        "blockSharpnessStd": block_sharpness_std,
    }


# ==========================================
# PROCESS ONE IMAGE
# ==========================================

def process_image(
    image_path,
    label
):

    mean_difference, max_difference, standard_deviation = calculate_ela(
        image_path
    )

    forensic = calculate_forensics(
        image_path
    )

    return {
        "filename": image_path.name,
        "label": label,
        "meanDifference": mean_difference,
        "maxDifference": max_difference,
        "standardDeviation": standard_deviation,
        "edgeDensity": forensic["edgeDensity"],
        "noiseMean": forensic["noiseMean"],
        "noiseStd": forensic["noiseStd"],
        "blurScore": forensic["blurScore"],
        "sharpnessScore": forensic["sharpnessScore"],
        "blockSharpnessAverage": forensic["blockSharpnessAverage"],
        "blockSharpnessStd": forensic["blockSharpnessStd"],
    }


# ==========================================
# MAIN
# ==========================================

results = []

image_extensions = [
    ".jpg",
    ".jpeg",
    ".png"
]

print("=" * 60)
print("IMAGEGUARD - EXTERNAL ADM FORENSIC EXTRACTION")
print("=" * 60)

print()
print("Processing REAL images...")

for image_path in sorted(REAL_DIR.iterdir()):

    if image_path.suffix.lower() not in image_extensions:
        continue

    try:

        result = process_image(
            image_path,
            "REAL"
        )

        results.append(result)

    except Exception as e:

        print(
            "Error:",
            image_path.name,
            e
        )

print("REAL completed:", 500)

print()
print("Processing AI images...")

for image_path in sorted(AI_DIR.iterdir()):

    if image_path.suffix.lower() not in image_extensions:
        continue

    try:

        result = process_image(
            image_path,
            "AI"
        )

        results.append(result)

    except Exception as e:

        print(
            "Error:",
            image_path.name,
            e
        )

print("AI completed:", 500)


# ==========================================
# SAVE
# ==========================================

fieldnames = [
    "filename",
    "label",
    "meanDifference",
    "maxDifference",
    "standardDeviation",
    "edgeDensity",
    "noiseMean",
    "noiseStd",
    "blurScore",
    "sharpnessScore",
    "blockSharpnessAverage",
    "blockSharpnessStd",
]

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


print()
print("=" * 60)
print("FORENSIC EXTRACTION COMPLETE")
print("=" * 60)

print("Total images:", len(results))

print()
print("REAL:", sum(
    1 for r in results
    if r["label"] == "REAL"
))

print("AI:", sum(
    1 for r in results
    if r["label"] == "AI"
))

print()
print("Output:")
print(OUTPUT_FILE)