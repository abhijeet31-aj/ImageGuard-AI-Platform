import os
import cv2

from cv.ela import (
    generate_difference_image,
    calculate_ela_metrics
)

from cv.forgery import analyze_forgery


TEST_DIR = r"D:\Users\ABHIJEET\Downloads\7th-sem-project\ImageGuard\python-engine\test_images"

TARGET_FILES = [
    "original2.jpg",
    "original3.jpg",
    "original4.jpg",
    "E-commerce1.jpg",
    "edited5.jpg"
]


def analyze_image(filename):

    image_path = os.path.join(
        TEST_DIR,
        filename
    )

    print()
    print("-" * 100)
    print(f"Filename : {filename}")

    if not os.path.exists(image_path):
        print("ERROR: Image not found")
        return

    image_cv = cv2.imread(image_path)

    if image_cv is None:
        print("ERROR: Could not read image")
        return

    # -----------------------------------------
    # ELA
    # -----------------------------------------

    with open(image_path, "rb") as file:
        image_bytes = file.read()

    difference_image = generate_difference_image(
        image_bytes
    )

    ela_metrics = calculate_ela_metrics(
        difference_image
    )

    # -----------------------------------------
    # OpenCV Forensic Analysis
    # -----------------------------------------

    forgery_analysis = analyze_forgery(
        image_cv
    )

    # -----------------------------------------
    # Results
    # -----------------------------------------

    print()
    print("ELA:")

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

    print()
    print("Forgery / OpenCV:")

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


def main():

    print("=" * 100)
    print("20-IMAGE ERROR / BORDERLINE FORENSIC ANALYSIS")
    print("=" * 100)

    print()
    print("Target images:")

    for filename in TARGET_FILES:
        print(f"  - {filename}")

    for filename in TARGET_FILES:
        analyze_image(filename)

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()