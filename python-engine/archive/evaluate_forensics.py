from pathlib import Path
import csv

from cv.image_reader import read_image
from cv.ela import (
    generate_difference_image,
    calculate_ela_metrics
)
from cv.forgery import analyze_forgery


DATASET_DIR = Path(
    r"D:\ImageGuardML\cas3120\processed\val"
)

OUTPUT_FILE = Path(
    "forensic_validation_results.csv"
)


def process_folder(folder, label):

    results = []

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() not in image_extensions:
            continue

        try:

            image_bytes = image_path.read_bytes()

            success, image_cv, image_info, error = read_image(
                image_bytes
            )

            if not success:
                print(
                    f"ERROR: {image_path.name} -> {error}"
                )
                continue

            # -------------------------
            # ELA
            # -------------------------

            difference_image = generate_difference_image(
                image_bytes
            )

            ela = calculate_ela_metrics(
                difference_image
            )

            # -------------------------
            # Forgery
            # -------------------------

            forgery = analyze_forgery(
                image_cv
            )

            results.append({

                "filename": image_path.name,

                "label": label,

                "meanDifference":
                    ela["meanDifference"],

                "maxDifference":
                    ela["maxDifference"],

                "standardDeviation":
                    ela["standardDeviation"],

                "edgeDensity":
                    forgery["edgeDensity"],

                "noiseMean":
                    forgery["noiseMean"],

                "noiseStd":
                    forgery["noiseStd"],

                "blurScore":
                    forgery["blurScore"],

                "sharpnessScore":
                    forgery["sharpnessScore"],

                "blockSharpnessAverage":
                    forgery["blockSharpnessAverage"],

                "blockSharpnessStd":
                    forgery["blockSharpnessStd"]

            })

        except Exception as error:

            print(
                f"ERROR: {image_path.name} -> {error}"
            )

    return results


print("=" * 60)
print("Collecting ELA + Forgery validation metrics")
print("=" * 60)


real_results = process_folder(
    DATASET_DIR / "0_real",
    "REAL"
)

fake_results = process_folder(
    DATASET_DIR / "1_fake",
    "AI"
)


results = real_results + fake_results


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
    "blockSharpnessStd"
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
print("DONE")
print("=" * 60)

print(f"REAL images: {len(real_results)}")
print(f"AI images  : {len(fake_results)}")
print(f"Total      : {len(results)}")

print(f"\nSaved to: {OUTPUT_FILE}")