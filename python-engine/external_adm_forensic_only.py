import csv
from pathlib import Path

import numpy as np


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FORENSIC_FILE = (
    BASE_DIR / "external_adm_forensic_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.50
FOLDS = 5
L2 = 0.01

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


# ============================================================
# LOAD DATA
# ============================================================

X = []
y = []
filenames = []


with open(
    FORENSIC_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        features = [
            float(row[feature])
            for feature in FEATURES
        ]

        label = (
            1
            if row["label"] == "AI"
            else 0
        )

        X.append(features)
        y.append(label)
        filenames.append(row["filename"])


X = np.array(
    X,
    dtype=np.float64
)

y = np.array(
    y,
    dtype=np.int64
)


# ============================================================
# CHECK DATA
# ============================================================

print("=" * 60)
print("IMAGEGUARD - EXTERNAL ADM FORENSIC ONLY")
print("=" * 60)

print()
print("Total samples:", len(X))
print("REAL:", np.sum(y == 0))
print("AI:", np.sum(y == 1))


if len(X) != 1000:
    raise ValueError(
        f"Expected 1000 samples, got {len(X)}"
    )


# ============================================================
# SIGMOID
# ============================================================

def sigmoid(z):

    z = np.clip(
        z,
        -50,
        50
    )

    return 1.0 / (
        1.0 + np.exp(-z)
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def train_logistic(
    X_train,
    y_train,
    l2=0.01,
    learning_rate=0.05,
    epochs=3000
):

    n_samples, n_features = X_train.shape

    weights = np.zeros(
        n_features
    )

    bias = 0.0

    for _ in range(epochs):

        logits = (
            X_train @ weights
            + bias
        )

        probabilities = sigmoid(
            logits
        )

        error = (
            probabilities
            - y_train
        )

        grad_w = (
            X_train.T @ error
        ) / n_samples

        grad_b = np.mean(error)

        grad_w += (
            l2 * weights
        )

        weights -= (
            learning_rate * grad_w
        )

        bias -= (
            learning_rate * grad_b
        )

    return weights, bias


# ============================================================
# STRATIFIED FOLDS
# ============================================================

def stratified_folds(
    labels,
    n_folds=5
):

    rng = np.random.default_rng(42)

    real_indices = np.where(
        labels == 0
    )[0]

    ai_indices = np.where(
        labels == 1
    )[0]

    rng.shuffle(real_indices)
    rng.shuffle(ai_indices)

    real_folds = np.array_split(
        real_indices,
        n_folds
    )

    ai_folds = np.array_split(
        ai_indices,
        n_folds
    )

    folds = []

    for i in range(n_folds):

        test_indices = np.concatenate([
            real_folds[i],
            ai_folds[i]
        ])

        train_real = np.concatenate([
            real_folds[j]
            for j in range(n_folds)
            if j != i
        ])

        train_ai = np.concatenate([
            ai_folds[j]
            for j in range(n_folds)
            if j != i
        ])

        train_indices = np.concatenate([
            train_real,
            train_ai
        ])

        folds.append(
            (
                train_indices,
                test_indices
            )
        )

    return folds


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize(
    X_train,
    X_test
):

    mean = np.mean(
        X_train,
        axis=0
    )

    std = np.std(
        X_train,
        axis=0
    )

    std[std == 0] = 1.0

    X_train_scaled = (
        X_train - mean
    ) / std

    X_test_scaled = (
        X_test - mean
    ) / std

    return (
        X_train_scaled,
        X_test_scaled
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

predictions = np.zeros(
    len(y),
    dtype=int
)

probabilities = np.zeros(
    len(y)
)


folds = stratified_folds(
    y,
    FOLDS
)


print()
print("=" * 60)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 60)


for fold_number, (
    train_idx,
    test_idx
) in enumerate(
    folds,
    start=1
):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]

    # Training-fold statistics only
    X_train_scaled, X_test_scaled = (
        standardize(
            X_train,
            X_test
        )
    )

    weights, bias = train_logistic(
        X_train_scaled,
        y_train,
        l2=L2
    )

    test_probabilities = sigmoid(
        X_test_scaled @ weights
        + bias
    )

    test_predictions = (
        test_probabilities >= THRESHOLD
    ).astype(int)

    predictions[test_idx] = (
        test_predictions
    )

    probabilities[test_idx] = (
        test_probabilities
    )

    print(
        f"Fold {fold_number}: "
        f"Train={len(train_idx)}, "
        f"Test={len(test_idx)}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

TP = int(np.sum(
    (y == 1) &
    (predictions == 1)
))

TN = int(np.sum(
    (y == 0) &
    (predictions == 0)
))

FP = int(np.sum(
    (y == 0) &
    (predictions == 1)
))

FN = int(np.sum(
    (y == 1) &
    (predictions == 0)
))


total = (
    TP + TN + FP + FN
)


# ============================================================
# METRICS
# ============================================================

accuracy = (
    (TP + TN) / total
)

precision = (
    TP / (TP + FP)
    if TP + FP > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if TP + FN > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0
)

fpr = (
    FP / (FP + TN)
    if FP + TN > 0
    else 0
)

fnr = (
    FN / (FN + TP)
    if FN + TP > 0
    else 0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("EXTERNAL ADM FORENSIC-ONLY RESULTS")
print("=" * 60)

print()
print("Dataset")
print("-" * 60)

print("REAL:", np.sum(y == 0))
print("AI:", np.sum(y == 1))
print("TOTAL:", total)

print()
print("Performance")
print("-" * 60)

print(
    f"Accuracy:            "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision:           "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall:              "
    f"{recall * 100:.2f}%"
)

print(
    f"F1-Score:            "
    f"{f1 * 100:.2f}%"
)

print(
    f"False Positive Rate: "
    f"{fpr * 100:.2f}%"
)

print(
    f"False Negative Rate: "
    f"{fnr * 100:.2f}%"
)

print()
print("Confusion Matrix")
print("-" * 60)

print(
    "                 Predicted"
)

print(
    "                 REAL    AI"
)

print(
    f"Actual REAL      "
    f"{TN:4d}   {FP:4d}"
)

print(
    f"Actual AI        "
    f"{FN:4d}   {TP:4d}"
)

print()
print("=" * 60)