import os
import cv2

from cv.ela import (
    generate_difference_image,
    calculate_ela_metrics
)

from cv.forgery import analyze_forgery


EDITED_DIR = r"D:\ImageGuardML\Community-Forensics\test_images"


def analyze_image(image_path):

    filename = os.path.basename(image_path)

    print("-" * 100)
    print(f"Filename : {filename}")

    image_cv = cv2.imread(image_path)

    if image_cv is None:
        print("ERROR: Could not read image")
        return

    # Read image bytes for ELA
    with open(image_path, "rb") as file:
        image_bytes = file.read()

    # -----------------------------
    # ELA
    # -----------------------------

    difference_image = generate_difference_image(
        image_bytes
    )

    ela_metrics = calculate_ela_metrics(
        difference_image
    )

    # -----------------------------
    # OpenCV Forgery Analysis
    # -----------------------------

    forgery_analysis = analyze_forgery(
        image_cv
    )

    # -----------------------------
    # Print Results
    # -----------------------------

    print("\nELA:")
    print(
        f"  Mean Difference    : "
        f"{ela_metrics['meanDifference']}"
    )

    print(
        f"  Max Difference     : "
        f"{ela_metrics['maxDifference']}"
    )

    print(
        f"  Standard Deviation : "
        f"{ela_metrics['standardDeviation']}"
    )

    print("\nForgery / OpenCV:")

    print(
        f"  Edge Density       : "
        f"{forgery_analysis['edgeDensity']}"
    )

    print(
        f"  Noise Mean         : "
        f"{forgery_analysis['noiseMean']}"
    )

    print(
        f"  Noise Std          : "
        f"{forgery_analysis['noiseStd']}"
    )

    print(
        f"  Blur Score         : "
        f"{forgery_analysis['blurScore']}"
    )

    print(
        f"  Sharpness Score    : "
        f"{forgery_analysis['sharpnessScore']}"
    )

    print(
        f"  Block Average      : "
        f"{forgery_analysis['blockSharpnessAverage']}"
    )

    print(
        f"  Block Std          : "
        f"{forgery_analysis['blockSharpnessStd']}"
    )

    print()


def main():

    print("=" * 100)
    print("EDITED IMAGE FORENSIC ANALYSIS")
    print("=" * 100)

    if not os.path.exists(EDITED_DIR):
        print("ERROR: Directory not found:")
        print(EDITED_DIR)
        return

    files = [
        f
        for f in os.listdir(EDITED_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
        and f.lower().startswith("edited")
    ]

    files.sort()

    print(f"\nFound {len(files)} edited images\n")

    for filename in files:

        image_path = os.path.join(
            EDITED_DIR,
            filename
        )

        analyze_image(image_path)

    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()