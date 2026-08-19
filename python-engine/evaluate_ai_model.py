from pathlib import Path
import csv

from ai.detector import detect_ai_image


VAL_DIR = Path(
    r"D:\ImageGuardML\cas3120\processed\val"
)

OUTPUT_FILE = Path(
    "ai_validation_predictions.csv"
)


def process_folder(folder, true_label):

    results = []

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() not in extensions:
            continue

        try:

            result = detect_ai_image(
                str(image_path)
            )

            prediction = result["prediction"]
            ai_probability = float(
                result["ai_probability"]
            )

            correct = (
                (
                    true_label == "AI"
                    and prediction == "AI Generated"
                )
                or
                (
                    true_label == "REAL"
                    and prediction == "Real"
                )
            )

            results.append({
                "filename": image_path.name,
                "true_label": true_label,
                "prediction": prediction,
                "ai_probability": ai_probability,
                "confidence": result["confidence"],
                "threshold": result["threshold"],
                "correct": correct
            })

            print(
                f"{image_path.name} | "
                f"TRUE={true_label} | "
                f"PRED={prediction} | "
                f"AI={ai_probability:.4f} | "
                f"CORRECT={correct}"
            )

        except Exception as error:

            print(
                f"{image_path.name} -> ERROR: {error}"
            )

    return results


print("=" * 70)
print("ImageGuardML VALIDATION ERROR ANALYSIS")
print("=" * 70)

real_results = process_folder(
    VAL_DIR / "0_real",
    "REAL"
)

ai_results = process_folder(
    VAL_DIR / "1_fake",
    "AI"
)

results = real_results + ai_results


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "filename",
        "true_label",
        "prediction",
        "ai_probability",
        "confidence",
        "threshold",
        "correct"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


total = len(results)
correct = sum(
    1 for row in results
    if row["correct"] == True
)

incorrect = total - correct

print()
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print(f"Total images : {total}")
print(f"Correct      : {correct}")
print(f"Incorrect    : {incorrect}")

if total:
    print(
        f"Accuracy     : "
        f"{correct / total * 100:.2f}%"
    )

print()
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 70)