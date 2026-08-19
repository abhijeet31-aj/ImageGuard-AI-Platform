import csv


AI_FILE = "ai_validation_predictions.csv"
FORENSIC_FILE = "forensic_validation_results.csv"


# --------------------------------------------------
# Load AI predictions
# --------------------------------------------------

with open(
    AI_FILE,
    encoding="utf-8"
) as file:

    ai_rows = list(
        csv.DictReader(file)
    )


# --------------------------------------------------
# Load forensic results
# --------------------------------------------------

with open(
    FORENSIC_FILE,
    encoding="utf-8"
) as file:

    forensic_rows = list(
        csv.DictReader(file)
    )


# --------------------------------------------------
# Create lookup
# --------------------------------------------------

forensic_lookup = {
    row["filename"]: row
    for row in forensic_rows
}


# --------------------------------------------------
# Find incorrect predictions
# --------------------------------------------------

wrong_predictions = [
    row
    for row in ai_rows
    if row["correct"].lower() == "false"
]


print()
print("=" * 100)
print("ERROR CASE FORENSIC ANALYSIS")
print("=" * 100)


for ai_row in wrong_predictions:

    filename = ai_row["filename"]

    forensic = forensic_lookup.get(
        filename
    )

    print()
    print("-" * 100)

    print(
        f"Filename       : {filename}"
    )

    print(
        f"True Label     : {ai_row['true_label']}"
    )

    print(
        f"Prediction     : {ai_row['prediction']}"
    )

    print(
        f"AI Probability : {float(ai_row['ai_probability']) * 100:.2f}%"
    )

    if forensic is None:

        print(
            "Forensic data  : NOT FOUND"
        )

        continue


    print()
    print("ELA:")
    print(
        f"  Mean Difference    : "
        f"{forensic['meanDifference']}"
    )

    print(
        f"  Max Difference     : "
        f"{forensic['maxDifference']}"
    )

    print(
        f"  Standard Deviation : "
        f"{forensic['standardDeviation']}"
    )


    print()
    print("Forgery:")
    print(
        f"  Edge Density       : "
        f"{forensic['edgeDensity']}"
    )

    print(
        f"  Noise Mean         : "
        f"{forensic['noiseMean']}"
    )

    print(
        f"  Noise Std          : "
        f"{forensic['noiseStd']}"
    )

    print(
        f"  Blur Score         : "
        f"{forensic['blurScore']}"
    )

    print(
        f"  Sharpness Score    : "
        f"{forensic['sharpnessScore']}"
    )

    print(
        f"  Block Average      : "
        f"{forensic['blockSharpnessAverage']}"
    )

    print(
        f"  Block Std          : "
        f"{forensic['blockSharpnessStd']}"
    )


print()
print("=" * 100)
print(
    f"Total error cases: {len(wrong_predictions)}"
)
print("=" * 100)