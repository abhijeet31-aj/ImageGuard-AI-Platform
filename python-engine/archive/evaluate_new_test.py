import os
import csv
from pathlib import Path

from ai.detector import detect_ai_image


TEST_DIR = Path(
    r"D:\Users\ABHIJEET\Downloads\7th-sem-project\ImageGuard\python-engine\new-test"
)

OUTPUT_FILE = "new_test_predictions.csv"


def get_true_label(filename):
    if filename.lower().startswith("ai"):
        return "AI"

    if filename.lower().startswith("original"):
        return "REAL"

    return None


image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


images = sorted(
    [
        p
        for p in TEST_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_extensions
    ],
    key=lambda p: p.name
)


print()
print("=" * 100)
print("IMAGEGUARD - INDEPENDENT 100 IMAGE TEST")
print("=" * 100)

print()
print(f"Total images found : {len(images)}")

results = []


for index, image_path in enumerate(images, start=1):

    true_label = get_true_label(
        image_path.name
    )

    if true_label is None:
        print(
            f"Skipping unknown file: {image_path.name}"
        )
        continue

    result = detect_ai_image(
        str(image_path)
    )

    ai_probability = float(
        result["ai_probability"]
    )

    prediction = (
        "AI"
        if result["prediction"] == "AI Generated"
        else "REAL"
    )

    correct = (
        prediction == true_label
    )

    results.append({
        "filename": image_path.name,
        "true_label": true_label,
        "prediction": prediction,
        "ai_probability": ai_probability,
        "confidence": result["confidence"],
        "correct": correct
    })

    print(
        f"[{index:3}/{len(images)}] "
        f"{image_path.name:<22} "
        f"True: {true_label:<4} "
        f"Pred: {prediction:<4} "
        f"AI: {ai_probability * 100:6.2f}% "
        f"{'OK' if correct else 'WRONG'}"
    )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

tp = sum(
    1 for r in results
    if r["true_label"] == "AI"
    and r["prediction"] == "AI"
)

tn = sum(
    1 for r in results
    if r["true_label"] == "REAL"
    and r["prediction"] == "REAL"
)

fp = sum(
    1 for r in results
    if r["true_label"] == "REAL"
    and r["prediction"] == "AI"
)

fn = sum(
    1 for r in results
    if r["true_label"] == "AI"
    and r["prediction"] == "REAL"
)


total = len(results)

accuracy = (
    (tp + tn) / total
    if total else 0
)

precision = (
    tp / (tp + fp)
    if (tp + fp) else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) else 0
)

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) else 0
)

false_negative_rate = (
    fn / (fn + tp)
    if (fn + tp) else 0
)


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "true_label",
            "prediction",
            "ai_probability",
            "confidence",
            "correct"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# Final report
# --------------------------------------------------

print()
print("=" * 100)
print("CONFUSION MATRIX")
print("=" * 100)

print(
    f"{'':20} {'Pred REAL':>15} {'Pred AI':>15}"
)

print(
    f"{'Actual REAL':20} "
    f"{tn:>15} "
    f"{fp:>15}"
)

print(
    f"{'Actual AI':20} "
    f"{fn:>15} "
    f"{tp:>15}"
)


print()
print("=" * 100)
print("METRICS")
print("=" * 100)

print(f"Total images          : {total}")
print(f"Correct predictions   : {tp + tn}")
print(f"Wrong predictions     : {fp + fn}")
print(f"Accuracy              : {accuracy * 100:.2f}%")
print(f"Precision             : {precision * 100:.2f}%")
print(f"Recall                : {recall * 100:.2f}%")
print(f"F1 Score              : {f1 * 100:.2f}%")
print(
    f"False Positive Rate   : "
    f"{false_positive_rate * 100:.2f}%"
)
print(
    f"False Negative Rate   : "
    f"{false_negative_rate * 100:.2f}%"
)


print()
print("=" * 100)
print("ERROR CASES")
print("=" * 100)

for result in results:

    if not result["correct"]:

        print(
            f"{result['filename']:<22} "
            f"Actual: {result['true_label']:<4} "
            f"Predicted: {result['prediction']:<4} "
            f"AI: {result['ai_probability'] * 100:.2f}%"
        )


print()
print("=" * 100)
print(
    f"Results saved to: {OUTPUT_FILE}"
)
print("=" * 100)