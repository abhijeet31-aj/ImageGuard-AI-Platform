import csv
from pathlib import Path

import cv2

from cv.ela import generate_difference_image, calculate_ela_metrics
from cv.forgery import analyze_forgery


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = Path(
    r"D:\Users\ABHIJEET\Downloads\7th-sem-project\ImageGuard\python-engine\new-test"
)

OUTPUT_FILE = "forensic_100_results.csv"


# ============================================================
# GET LABEL FROM FILENAME
# ============================================================

def get_label(filename):

    name = filename.lower()

    if name.startswith("ai"):
        return "AI"

    if name.startswith("original"):
        return "REAL"

    return None


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(image_path):

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # --------------------------------------------------------
    # ELA
    # --------------------------------------------------------

    difference_image = generate_difference_image(
        image_bytes
    )

    ela_metrics = calculate_ela_metrics(
        difference_image
    )

    # --------------------------------------------------------
    # OpenCV
    # --------------------------------------------------------

    image_cv = cv2.imread(
        str(image_path)
    )

    if image_cv is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    forgery_metrics = analyze_forgery(
        image_cv
    )

    return {
        **ela_metrics,
        **forgery_metrics
    }


# ============================================================
# MAIN
# ============================================================

def main():

    image_files = sorted(
        [
            p
            for p in IMAGE_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in [".jpg", ".jpeg", ".png", ".webp"]
        ]
    )

    print("=" * 100)
    print("IMAGEGUARD - 100 IMAGE FORENSIC ANALYSIS")
    print("=" * 100)

    print()
    print(f"Image directory : {IMAGE_DIR}")
    print(f"Total images    : {len(image_files)}")

    if len(image_files) != 100:
        print()
        print(
            f"WARNING: Expected 100 images, "
            f"but found {len(image_files)}."
        )

    results = []

    print()

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        true_label = get_label(
            image_path.name
        )

        if true_label is None:
            print(
                f"[SKIP] Unknown label: "
                f"{image_path.name}"
            )
            continue

        try:

            metrics = analyze_image(
                image_path
            )

            row = {
                "filename": image_path.name,
                "true_label": true_label,

                "meanDifference":
                    metrics["meanDifference"],

                "maxDifference":
                    metrics["maxDifference"],

                "standardDeviation":
                    metrics["standardDeviation"],

                "edgeDensity":
                    metrics["edgeDensity"],

                "noiseMean":
                    metrics["noiseMean"],

                "noiseStd":
                    metrics["noiseStd"],

                "blurScore":
                    metrics["blurScore"],

                "sharpnessScore":
                    metrics["sharpnessScore"],

                "blockSharpnessAverage":
                    metrics["blockSharpnessAverage"],

                "blockSharpnessStd":
                    metrics["blockSharpnessStd"],
            }

            results.append(row)

            print(
                f"[{index:3}/100] "
                f"{image_path.name:<22} "
                f"{true_label:<4} "
                f"ELA={metrics['meanDifference']:<6} "
                f"Edge={metrics['edgeDensity']:<6}"
            )

        except Exception as e:

            print(
                f"[ERROR] "
                f"{image_path.name}: {e}"
            )


    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "filename",
        "true_label",

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
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


    # ========================================================
    # SUMMARY
    # ========================================================

    ai_rows = [
        r for r in results
        if r["true_label"] == "AI"
    ]

    real_rows = [
        r for r in results
        if r["true_label"] == "REAL"
    ]

    print()
    print("=" * 100)
    print("FORENSIC ANALYSIS COMPLETE")
    print("=" * 100)

    print()
    print(f"Total analyzed : {len(results)}")
    print(f"AI images      : {len(ai_rows)}")
    print(f"REAL images    : {len(real_rows)}")

    print()
    print(f"CSV saved to   : {OUTPUT_FILE}")

    print("=" * 100)


if __name__ == "__main__":
    main()