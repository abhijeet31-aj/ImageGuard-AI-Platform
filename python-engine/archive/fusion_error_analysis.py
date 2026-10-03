import csv
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_FILE = BASE_DIR / "ai_validation_predictions.csv"
FORENSIC_FILE = BASE_DIR / "forensic_validation_results.csv"

FORENSIC_FEATURES = [
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

THRESHOLD = 0.50


def load_rows(path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sigmoid(values):
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic_regression(features, labels, regularization, iterations=4000):
    weights = np.zeros(features.shape[1] + 1)

    design = np.column_stack([
        np.ones(len(features)),
        features
    ])

    penalty = np.r_[
        0.0,
        np.full(features.shape[1], regularization)
    ]

    for _ in range(iterations):
        probabilities = sigmoid(design @ weights)

        gradient = (
            design.T @ (probabilities - labels)
        ) / len(labels)

        gradient += penalty * weights / len(labels)

        weights -= 0.05 * gradient

    return weights


def stratified_folds(labels, folds=5):
    rng = np.random.default_rng(42)

    assignment = np.empty(
        len(labels),
        dtype=int
    )

    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        assignment[indices] = (
            np.arange(len(indices)) % folds
        )

    return assignment


def cross_validated_probabilities(
    features,
    labels,
    regularization
):
    folds = stratified_folds(labels)

    probabilities = np.empty(len(labels))

    for fold in range(5):

        train = folds != fold
        test = ~train

        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0)
        std[std == 0] = 1.0

        normalized = (
            features - mean
        ) / std

        weights = fit_logistic_regression(
            normalized[train],
            labels[train],
            regularization
        )

        test_design = np.column_stack([
            np.ones(test.sum()),
            normalized[test]
        ])

        probabilities[test] = sigmoid(
            test_design @ weights
        )

    return probabilities


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

prediction_rows = load_rows(PREDICTIONS_FILE)
forensic_rows = load_rows(FORENSIC_FILE)

prediction_lookup = {
    (
        row["filename"],
        row["true_label"]
    ): row
    for row in prediction_rows
}

forensic_lookup = {
    (
        row["filename"],
        row["label"]
    ): row
    for row in forensic_rows
}

keys = sorted(forensic_lookup)

labels = np.array([
    int(label.upper() == "AI")
    for filename, label in keys
])

detector_probabilities = np.array([
    float(
        prediction_lookup[key]["ai_probability"]
    )
    for key in keys
])


# --------------------------------------------------
# FUSION FEATURES
# --------------------------------------------------

clipped = np.clip(
    detector_probabilities,
    1e-4,
    1 - 1e-4
)

detector_log_odds = np.log(
    clipped / (1 - clipped)
)

forensic_matrix = np.array([
    [
        float(
            forensic_lookup[key][feature]
        )
        for feature in FORENSIC_FEATURES
    ]
    for key in keys
])

fusion_matrix = np.column_stack([
    detector_log_odds,
    forensic_matrix
])


# --------------------------------------------------
# SAME VALIDATION PROCEDURE
# --------------------------------------------------

probabilities = cross_validated_probabilities(
    fusion_matrix,
    labels,
    regularization=0.01
)

predicted = probabilities >= THRESHOLD


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

TN = int(((labels == 0) & (predicted == 0)).sum())
FP = int(((labels == 0) & (predicted == 1)).sum())
FN = int(((labels == 1) & (predicted == 0)).sum())
TP = int(((labels == 1) & (predicted == 1)).sum())

accuracy = (TN + TP) / len(labels)

precision = (
    TP / (TP + FP)
    if TP + FP else 0
)

recall = (
    TP / (TP + FN)
    if TP + FN else 0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if precision + recall else 0
)


print("=" * 90)
print("IMAGEGUARD - 1000 IMAGE FUSION ERROR ANALYSIS")
print("=" * 90)

print(f"Threshold : {THRESHOLD:.2f}")
print(f"Images    : {len(labels)}")
print()

print("CONFUSION MATRIX")
print("-" * 60)
print(f"                 Predicted REAL    Predicted AI")
print(f"Actual REAL          {TN:<15}{FP}")
print(f"Actual AI            {FN:<15}{TP}")

print()
print("METRICS")
print("-" * 60)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print()
print("=" * 90)
print("ERROR CASES")
print("=" * 90)

errors = []

for i, key in enumerate(keys):

    if labels[i] != int(predicted[i]):

        filename, true_label = key

        errors.append({
            "filename": filename,
            "true_label": true_label,
            "ai_probability": detector_probabilities[i],
            "fusion_probability": probabilities[i],
            "meanDifference": forensic_matrix[i][0],
            "maxDifference": forensic_matrix[i][1],
            "standardDeviation": forensic_matrix[i][2],
            "edgeDensity": forensic_matrix[i][3],
            "noiseMean": forensic_matrix[i][4],
            "noiseStd": forensic_matrix[i][5],
            "blurScore": forensic_matrix[i][6],
            "sharpnessScore": forensic_matrix[i][7],
            "blockSharpnessAverage": forensic_matrix[i][8],
            "blockSharpnessStd": forensic_matrix[i][9],
})

        print()
        print(f"Filename          : {filename}")
        print(f"True Label        : {true_label}")
        print(
            f"AI Probability    : "
            f"{detector_probabilities[i] * 100:.2f}%"
        )
        print(
            f"Fusion Probability: "
            f"{probabilities[i] * 100:.2f}%"
        )
        print(
            f"Prediction        : "
            f"{'AI Generated' if predicted[i] else 'Real'}"
        )
        print("Forensic Features")
        print(f"  meanDifference          : {forensic_matrix[i][0]:.2f}")
        print(f"  maxDifference           : {forensic_matrix[i][1]:.2f}")
        print(f"  standardDeviation       : {forensic_matrix[i][2]:.2f}")
        print(f"  edgeDensity             : {forensic_matrix[i][3]:.2f}")
        print(f"  noiseMean               : {forensic_matrix[i][4]:.2f}")
        print(f"  noiseStd                : {forensic_matrix[i][5]:.2f}")
        print(f"  blurScore               : {forensic_matrix[i][6]:.2f}")
        print(f"  sharpnessScore          : {forensic_matrix[i][7]:.2f}")
        print(f"  blockSharpnessAverage   : {forensic_matrix[i][8]:.2f}")
        print(f"  blockSharpnessStd       : {forensic_matrix[i][9]:.2f}")

print()
print("-" * 90)
print(f"Total errors: {len(errors)}")

ai_to_real = sum(
    1
    for e in errors
    if e["true_label"].upper() == "AI"
)

real_to_ai = sum(
    1
    for e in errors
    if e["true_label"].upper() == "REAL"
)

print(f"AI -> REAL errors : {ai_to_real}")
print(f"REAL -> AI errors : {real_to_ai}")

print("=" * 90)
print("ERROR ANALYSIS COMPLETE")
print("=" * 90)