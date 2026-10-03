import csv


CSV_FILE = "ai_validation_predictions.csv"


# --------------------------------------------------
# Load validation predictions
# --------------------------------------------------

with open(CSV_FILE, encoding="utf-8") as file:
    rows = list(csv.DictReader(file))


# --------------------------------------------------
# Probability bins
# --------------------------------------------------

bins = [
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.01),
]


print()
print("=" * 110)
print("AI PROBABILITY CALIBRATION ANALYSIS")
print("=" * 110)

print()

header = (
    "RANGE".ljust(14)
    + "TOTAL".rjust(8)
    + "REAL".rjust(8)
    + "AI".rjust(8)
    + "AI %".rjust(10)
    + "REAL %".rjust(10)
    + "AVG PROB".rjust(12)
)

print(header)
print("-" * 110)


for lower, upper in bins:

    bin_rows = []

    for row in rows:

        probability = float(
            row["ai_probability"]
        )

        if lower <= probability < upper:
            bin_rows.append(row)


    total = len(bin_rows)

    if total == 0:
        continue


    real_count = sum(
        row["true_label"] == "REAL"
        for row in bin_rows
    )

    ai_count = sum(
        row["true_label"] == "AI"
        for row in bin_rows
    )


    ai_percentage = (
        ai_count / total * 100
    )

    real_percentage = (
        real_count / total * 100
    )


    average_probability = sum(
        float(row["ai_probability"])
        for row in bin_rows
    ) / total


    range_text = (
        f"{lower:.1f}-{min(upper, 1.0):.1f}"
    )


    print(
        range_text.ljust(14)
        + str(total).rjust(8)
        + str(real_count).rjust(8)
        + str(ai_count).rjust(8)
        + f"{ai_percentage:.2f}%".rjust(10)
        + f"{real_percentage:.2f}%".rjust(10)
        + f"{average_probability:.4f}".rjust(12)
    )


# --------------------------------------------------
# Overall calibration information
# --------------------------------------------------

print()
print("=" * 110)
print("HIGH-PROBABILITY ANALYSIS")
print("=" * 110)


for threshold in [0.35, 0.50, 0.70, 0.90]:

    selected = [
        row
        for row in rows
        if float(row["ai_probability"]) >= threshold
    ]


    total = len(selected)

    real_count = sum(
        row["true_label"] == "REAL"
        for row in selected
    )

    ai_count = sum(
        row["true_label"] == "AI"
        for row in selected
    )


    if total == 0:
        continue


    print()
    print(
        f"Threshold >= {threshold:.2f}"
    )

    print(
        f"Total images : {total}"
    )

    print(
        f"REAL         : {real_count}"
    )

    print(
        f"AI           : {ai_count}"
    )

    print(
        f"Actual AI %  : {ai_count / total * 100:.2f}%"
    )

    print(
        f"Actual REAL %: {real_count / total * 100:.2f}%"
    )


print()
print("=" * 110)
print("CALIBRATION ANALYSIS COMPLETE")
print("=" * 110)