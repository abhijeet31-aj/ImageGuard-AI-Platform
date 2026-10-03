"""Read-only validation experiment for detector-score and forensic-feature fusion.

This script does not load, retrain, or alter the production image detector.  It
uses the saved validation predictions and forensic CSV only, and evaluates a
small logistic-regression meta-model with stratified five-fold cross-validation.
"""

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


def roc_auc(labels, scores):
    """Tie-aware ROC AUC without an additional dependency."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0

    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positives = labels.sum()
    negatives = len(labels) - positives
    return (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def fit_logistic_regression(features, labels, regularization, iterations=4000):
    """Fit a small L2-regularized logistic model; intercept is not penalized."""
    weights = np.zeros(features.shape[1] + 1)
    design = np.column_stack([np.ones(len(features)), features])
    penalty = np.r_[0.0, np.full(features.shape[1], regularization)]

    for _ in range(iterations):
        probabilities = sigmoid(design @ weights)
        gradient = (design.T @ (probabilities - labels)) / len(labels)
        gradient += penalty * weights / len(labels)
        weights -= 0.05 * gradient

    return weights


def stratified_folds(labels, folds=5):
    rng = np.random.default_rng(42)
    assignment = np.empty(len(labels), dtype=int)

    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        assignment[indices] = np.arange(len(indices)) % folds

    return assignment


def cross_validated_probabilities(features, labels, regularization):
    folds = stratified_folds(labels)
    probabilities = np.empty(len(labels))

    for fold in range(5):
        train = folds != fold
        test = ~train
        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0)
        std[std == 0] = 1.0
        normalized = (features - mean) / std
        weights = fit_logistic_regression(
            normalized[train], labels[train], regularization
        )
        probabilities[test] = sigmoid(
            np.column_stack([np.ones(test.sum()), normalized[test]]) @ weights
        )

    return probabilities


def metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    true_negative = int(((labels == 0) & ~predicted).sum())
    false_positive = int(((labels == 0) & predicted).sum())
    false_negative = int(((labels == 1) & ~predicted).sum())
    true_positive = int(((labels == 1) & predicted).sum())
    accuracy = (true_negative + true_positive) / len(labels)
    balanced_accuracy = (
        true_negative / (true_negative + false_positive)
        + true_positive / (true_positive + false_negative)
    ) / 2
    return accuracy, balanced_accuracy, true_negative, false_positive, false_negative, true_positive


prediction_rows = load_rows(PREDICTIONS_FILE)
forensic_rows = load_rows(FORENSIC_FILE)

prediction_lookup = {
    (row["filename"], row["true_label"]): row for row in prediction_rows
}
forensic_lookup = {
    (row["filename"], row["label"]): row for row in forensic_rows
}

if len(prediction_lookup) != len(prediction_rows):
    raise ValueError("Prediction rows are not unique by (filename, true_label).")
if len(forensic_lookup) != len(forensic_rows):
    raise ValueError("Forensic rows are not unique by (filename, label).")
if set(prediction_lookup) != set(forensic_lookup):
    raise ValueError("Prediction and forensic CSVs do not contain the same keyed rows.")

keys = sorted(forensic_lookup)
labels = np.array([int(label == "AI") for _, label in keys])
detector_probabilities = np.array(
    [float(prediction_lookup[key]["ai_probability"]) for key in keys]
)

# Log-odds preserves the detector score while making it suitable for a linear
# fusion model.  Clipping avoids infinite values from recorded 0/1 scores.
clipped_probability = np.clip(detector_probabilities, 1e-4, 1 - 1e-4)
detector_log_odds = np.log(clipped_probability / (1 - clipped_probability))
forensic_matrix = np.array(
    [
        [float(forensic_lookup[key][feature]) for feature in FORENSIC_FEATURES]
        for key in keys
    ]
)
fusion_matrix = np.column_stack([detector_log_odds, forensic_matrix])

baseline = metrics(labels, detector_probabilities, threshold=0.35)
print("=" * 88)
print("READ-ONLY DETECTOR + FORENSIC FUSION EXPERIMENT")
print("=" * 88)
print(f"Matched validation examples: {len(keys)}")
print("Match key: filename + true label")
print()
print("Baseline detector (recorded threshold: 0.35)")
print(
    "Accuracy: {:.2%} | Balanced accuracy: {:.2%} | TN={} FP={} FN={} TP={}".format(
        *baseline
    )
)
print()

best = None
for regularization in (0.01, 0.1, 1.0, 10.0, 100.0):
    probabilities = cross_validated_probabilities(
        fusion_matrix, labels, regularization
    )
    result = metrics(labels, probabilities, threshold=0.5)
    candidate = (result[0], regularization, probabilities, result)
    if best is None or candidate[0] > best[0]:
        best = candidate

_, regularization, probabilities, fusion = best
print("Fusion: detector log-odds + 10 forensic features")
print("Evaluation: stratified 5-fold cross-validation; logistic regression")
print(f"Selected L2 regularization: {regularization}")
print(
    "Accuracy: {:.2%} | Balanced accuracy: {:.2%} | TN={} FP={} FN={} TP={}".format(
        *fusion
    )
)
print(f"Cross-validated ROC-AUC: {roc_auc(labels, probabilities):.4f}")
print()

if fusion[0] > baseline[0]:
    print("RESULT: Fusion improved this exploratory cross-validated comparison.")
    print("Do not deploy yet: confirm once on a new, untouched test dataset.")
else:
    print("RESULT: Fusion did not beat the current detector on this validation set.")
    print("Recommendation: do not add forensic features to the production decision.")
