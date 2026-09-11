import csv

rows = list(csv.DictReader(open("new_test_predictions.csv", encoding="utf-8")))

print("=" * 75)
print("IMAGEGUARD - AI THRESHOLD ANALYSIS")
print("=" * 75)
print(f"{'THRESHOLD':<12}{'ACC':>10}{'PRECISION':>12}{'RECALL':>10}{'F1':>10}{'FPR':>10}{'FNR':>10}")
print("-" * 75)

for threshold in [i / 100 for i in range(10, 91, 5)]:

    TP = TN = FP = FN = 0

    for r in rows:
        true_ai = r["true_label"].upper() == "AI"
        probability = float(r["ai_probability"])
        predicted_ai = probability >= threshold

        if true_ai and predicted_ai:
            TP += 1
        elif not true_ai and not predicted_ai:
            TN += 1
        elif not true_ai and predicted_ai:
            FP += 1
        elif true_ai and not predicted_ai:
            FN += 1

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = FP / (FP + TN) if (FP + TN) else 0
    fnr = FN / (FN + TP) if (FN + TP) else 0

    print(
        f"{threshold:<12.2f}"
        f"{accuracy*100:>9.2f}%"
        f"{precision*100:>11.2f}%"
        f"{recall*100:>9.2f}%"
        f"{f1*100:>9.2f}%"
        f"{fpr*100:>9.2f}%"
        f"{fnr*100:>9.2f}%"
    )

print("=" * 75)
