import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_FILE = BASE_DIR / "external_adm_predictions.csv"

TP = 0
TN = 0
FP = 0
FN = 0

with open(PREDICTIONS_FILE, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        true_label = row["true_label"]
        prediction = row["prediction"]

        if true_label == "AI" and prediction == "AI":
            TP += 1

        elif true_label == "REAL" and prediction == "REAL":
            TN += 1

        elif true_label == "REAL" and prediction == "AI":
            FP += 1

        elif true_label == "AI" and prediction == "REAL":
            FN += 1


total = TP + TN + FP + FN

accuracy = (TP + TN) / total

precision = TP / (TP + FP) if (TP + FP) else 0

recall = TP / (TP + FN) if (TP + FN) else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall)
    else 0
)

fpr = FP / (FP + TN) if (FP + TN) else 0

fnr = FN / (FN + TP) if (FN + TP) else 0


print("=" * 60)
print("IMAGEGUARD - EXTERNAL ADM EVALUATION")
print("=" * 60)

print()
print("Dataset")
print("REAL:", TN + FP)
print("AI:", TP + FN)
print("TOTAL:", total)

print()
print("Performance")
print("-" * 60)

print(f"Accuracy:            {accuracy * 100:.2f}%")
print(f"Precision:           {precision * 100:.2f}%")
print(f"Recall:              {recall * 100:.2f}%")
print(f"F1-Score:            {f1 * 100:.2f}%")
print(f"False Positive Rate: {fpr * 100:.2f}%")
print(f"False Negative Rate: {fnr * 100:.2f}%")

print()
print("Confusion Matrix")
print("-" * 60)

print("                 Predicted")
print("                 REAL    AI")
print(f"Actual REAL      {TN:4d}   {FP:4d}")
print(f"Actual AI        {FN:4d}   {TP:4d}")

print("=" * 60)