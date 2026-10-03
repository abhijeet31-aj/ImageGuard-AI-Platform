import csv

with open(
    "ai_validation_predictions.csv",
    encoding="utf-8"
) as file:

    rows = list(csv.DictReader(file))


wrong = [
    row
    for row in rows
    if row["correct"].lower() == "false"
]


print()
print("=" * 100)
print("WRONG PREDICTIONS")
print("=" * 100)

for row in wrong:

    print(
        f'{row["filename"]} | '
        f'True: {row["true_label"]} | '
        f'Predicted: {row["prediction"]} | '
        f'AI: {float(row["ai_probability"]) * 100:.2f}%'
    )

print()
print("=" * 100)
print(f"Total wrong predictions: {len(wrong)}")
print("=" * 100)