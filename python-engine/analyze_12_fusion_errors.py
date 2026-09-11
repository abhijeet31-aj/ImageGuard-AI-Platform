import csv

AI_FILE = "ai_validation_predictions.csv"
FORENSIC_FILE = "forensic_validation_results.csv"

ERROR_FILES = {
    "00001.png",
    "00202.png",
    "00213.png",
    "00255.png",
    "00260.png",
    "00278.png",
    "00309.png",
    "00313.png",
    "00332.png",
    "00369.png",
    "00397.png",
    "00406.png",
}

FEATURES = [
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


def load_csv(filename):
    with open(filename, encoding="utf-8") as f:
        return list(csv.DictReader(f))


ai_rows = load_csv(AI_FILE)
forensic_rows = load_csv(FORENSIC_FILE)

ai_lookup = {
    row["filename"]: row
    for row in ai_rows
}

forensic_lookup = {
    row["filename"]: row
    for row in forensic_rows
}


print("=" * 120)
print("IMAGEGUARD - 12 FUSION ERROR FORENSIC COMPARISON")
print("=" * 120)

print()

for filename in sorted(ERROR_FILES):

    ai = ai_lookup.get(filename)
    forensic = forensic_lookup.get(filename)

    if not ai:
        print(f"WARNING: AI result not found: {filename}")
        continue

    if not forensic:
        print(f"WARNING: Forensic result not found: {filename}")
        continue

    true_label = ai["true_label"]
    ai_probability = float(ai["ai_probability"])

    print("-" * 120)
    print(f"Filename          : {filename}")
    print(f"True Label        : {true_label}")
    print(f"AI Probability    : {ai_probability * 100:.2f}%")

    print()
    print("FORENSIC FEATURES")

    for feature in FEATURES:

        value = float(forensic[feature])

        print(
            f"{feature:<25}: {value:.2f}"
        )

print()
print("=" * 120)
print("ERROR GROUP SUMMARY")
print("=" * 120)

ai_to_real = []
real_to_ai = []

for filename in sorted(ERROR_FILES):

    ai = ai_lookup.get(filename)

    if not ai:
        continue

    if ai["true_label"].upper() == "AI":
        ai_to_real.append(filename)
    else:
        real_to_ai.append(filename)

print()
print(f"AI -> REAL errors : {len(ai_to_real)}")
print(f"REAL -> AI errors : {len(real_to_ai)}")

print()
print("AI -> REAL:")
for filename in ai_to_real:
    print(f"  {filename}")

print()
print("REAL -> AI:")
for filename in real_to_ai:
    print(f"  {filename}")

print()
print("=" * 120)
print("ANALYSIS COMPLETE")
print("=" * 120)