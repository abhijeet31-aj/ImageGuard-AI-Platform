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

        gradient += (
            penalty * weights / len(labels)
        )

        weights -= 0.05 * gradient

    return weights


def stratified_folds(labels, folds=5):
    rng = np.random.default_rng(42)

    assignment = np.empty(
        len(labels),
        dtype=int
    )

    for label in (0, 1):
        indices = np.flatnonzero(
            labels == label
        )

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

    probabilities = np.empty(
        len(labels)
    )

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


def calculate_metrics(
    labels,
    probabilities,
    threshold
):

    predicted = probabilities >= threshold

    TN = int(
        ((labels == 0) & ~predicted).sum()
    )

    FP = int(
        ((labels == 0) & predicted).sum()
    )

    FN = int(
        ((labels == 1) & ~predicted).sum()
    )

    TP = int(
        ((labels == 1) & predicted).sum()
    )

    accuracy = (
        (TN + TP) / len(labels)
    )

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

    fpr = (
        FP / (FP + TN)
        if FP + TN else 0
    )

    fnr = (
        FN / (FN + TP)
        if FN + TP else 0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        fpr,
        fnr,
        TN,
        FP,
        FN,
        TP
    )


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

prediction_rows = load_rows(
    PREDICTIONS_FILE
)

forensic_rows = load_rows(
    FORENSIC_FILE
)

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


if set(prediction_lookup) != set(forensic_lookup):
    raise ValueError(
        "Prediction and forensic CSVs do not contain the same images."
    )


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
# AI PROBABILITY → LOG ODDS
# --------------------------------------------------

clipped_probability = np.clip(
    detector_probabilities,
    1e-4,
    1 - 1e-4
)

detector_log_odds = np.log(
    clipped_probability
    / (1 - clipped_probability)
)


# --------------------------------------------------
# FORENSIC FEATURES
# --------------------------------------------------

forensic_matrix = np.array([
    [
        float(
            forensic_lookup[key][feature]
        )
        for feature in FORENSIC_FEATURES
    ]
    for key in keys
])


# --------------------------------------------------
# FUSION MATRIX
# --------------------------------------------------

fusion_matrix = np.column_stack([
    detector_log_odds,
    forensic_matrix
])


# --------------------------------------------------
# FIND BEST REGULARIZATION
# --------------------------------------------------

best = None

for regularization in (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
):

    probabilities = (
        cross_validated_probabilities(
            fusion_matrix,
            labels,
            regularization
        )
    )

    metrics = calculate_metrics(
        labels,
        probabilities,
        threshold=0.50
    )

    accuracy = metrics[0]

    if best is None or accuracy > best[0]:

        best = (
            accuracy,
            regularization,
            probabilities
        )


_, best_regularization, fusion_probabilities = best


# --------------------------------------------------
# THRESHOLD SWEEP
# --------------------------------------------------

print("=" * 95)
print("IMAGEGUARD - FUSION THRESHOLD ANALYSIS")
print("=" * 95)

print(
    f"Images: {len(labels)}"
)

print(
    f"Selected L2 regularization: "
    f"{best_regularization}"
)

print()

print(
    f"{'THRESHOLD':<12}"
    f"{'ACCURACY':>12}"
    f"{'PRECISION':>12}"
    f"{'RECALL':>10}"
    f"{'F1':>10}"
    f"{'FPR':>10}"
    f"{'FNR':>10}"
)

print("-" * 95)


results = []

for threshold in [
    i / 100
    for i in range(10, 91, 5)
]:

    result = calculate_metrics(
        labels,
        fusion_probabilities,
        threshold
    )

    results.append(
        (
            threshold,
            result
        )
    )

    (
        accuracy,
        precision,
        recall,
        f1,
        fpr,
        fnr,
        TN,
        FP,
        FN,
        TP
    ) = result

    print(
        f"{threshold:<12.2f}"
        f"{accuracy * 100:>11.2f}%"
        f"{precision * 100:>11.2f}%"
        f"{recall * 100:>9.2f}%"
        f"{f1 * 100:>9.2f}%"
        f"{fpr * 100:>9.2f}%"
        f"{fnr * 100:>9.2f}%"
    )


# --------------------------------------------------
# BEST THRESHOLDS
# --------------------------------------------------

best_accuracy = max(
    results,
    key=lambda x: x[1][0]
)

best_f1 = max(
    results,
    key=lambda x: x[1][3]
)

print()
print("=" * 95)
print("BEST RESULTS")
print("=" * 95)

print(
    f"Best Accuracy threshold : "
    f"{best_accuracy[0]:.2f} "
    f"({best_accuracy[1][0] * 100:.2f}%)"
)

print(
    f"Best F1 threshold       : "
    f"{best_f1[0]:.2f} "
    f"({best_f1[1][3] * 100:.2f}%)"
)

print("=" * 95)