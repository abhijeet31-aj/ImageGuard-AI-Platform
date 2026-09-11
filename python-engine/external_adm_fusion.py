import csv
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AI_FILE = BASE_DIR / "external_adm_predictions.csv"
FORENSIC_FILE = BASE_DIR / "external_adm_forensic_results.csv"


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.50
FOLDS = 5
L2 = 0.01

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


# ============================================================
# LOAD AI PREDICTIONS
# ============================================================

ai_data = {}

with open(AI_FILE, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        filename = row["filename"]

        ai_data[filename] = {
            "ai_probability": float(row["ai_probability"]),
            "true_label": 1 if row["true_label"] == "AI" else 0,
        }


# ============================================================
# LOAD FORENSIC FEATURES
# ============================================================

forensic_data = {}

with open(FORENSIC_FILE, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        filename = row["filename"]

        forensic_data[filename] = {
            feature: float(row[feature])
            for feature in FORENSIC_FEATURES
        }

        forensic_data[filename]["true_label"] = (
            1 if row["label"] == "AI" else 0
        )


# ============================================================
# IMPORTANT
# ============================================================
#
# The filenames are identical between REAL and AI folders.
# Therefore the filename alone cannot be used as the key.
#
# The extraction script used:
#
# REAL: real_0001.jpg ... real_0500.jpg
# AI:   ai_0001.jpg   ... ai_0500.jpg
#
# So these are unique and can safely be matched.
# ============================================================


# ============================================================
# BUILD DATASET
# ============================================================

X = []
y = []
ai_probabilities = []
filenames = []


for filename, ai_row in ai_data.items():

    if filename not in forensic_data:
        continue

    forensic_row = forensic_data[filename]

    features = [
        ai_row["ai_probability"]
    ]

    features.extend(
        forensic_row[feature]
        for feature in FORENSIC_FEATURES
    )

    X.append(features)
    y.append(ai_row["true_label"])
    ai_probabilities.append(ai_row["ai_probability"])
    filenames.append(filename)


X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.int64)
ai_probabilities = np.array(
    ai_probabilities,
    dtype=np.float64
)


# ============================================================
# CHECK DATA
# ============================================================

print("=" * 60)
print("IMAGEGUARD - EXTERNAL ADM FUSION")
print("=" * 60)

print()
print("AI records:", len(ai_data))
print("Forensic records:", len(forensic_data))
print("Matched records:", len(X))

if len(X) != 1000:

    raise ValueError(
        f"Expected 1000 matched samples, got {len(X)}"
    )

print("REAL:", np.sum(y == 0))
print("AI:", np.sum(y == 1))


# ============================================================
# LOGISTIC REGRESSION IMPLEMENTATION
# ============================================================

def sigmoid(z):

    z = np.clip(z, -50, 50)

    return 1.0 / (1.0 + np.exp(-z))


def train_logistic(
    X_train,
    y_train,
    l2=0.01,
    learning_rate=0.05,
    epochs=3000
):

    n_samples, n_features = X_train.shape

    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):

        logits = (
            X_train @ weights
            + bias
        )

        probabilities = sigmoid(logits)

        error = probabilities - y_train

        grad_w = (
            X_train.T @ error
        ) / n_samples

        grad_b = np.mean(error)

        grad_w += (
            l2 * weights
        )

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias


# ============================================================
# STRATIFIED 5-FOLD SPLIT
# ============================================================

def stratified_folds(y, n_folds=5):

    rng = np.random.default_rng(42)

    class_zero = np.where(y == 0)[0]
    class_one = np.where(y == 1)[0]

    rng.shuffle(class_zero)
    rng.shuffle(class_one)

    zero_folds = np.array_split(
        class_zero,
        n_folds
    )

    one_folds = np.array_split(
        class_one,
        n_folds
    )

    folds = []

    for i in range(n_folds):

        test_indices = np.concatenate([
            zero_folds[i],
            one_folds[i]
        ])

        train_indices = np.concatenate([
            np.concatenate([
                zero_folds[j]
                for j in range(n_folds)
                if j != i
            ]),
            np.concatenate([
                one_folds[j]
                for j in range(n_folds)
                if j != i
            ])
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

def standardize_train_test(
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
# CROSS-VALIDATION
# ============================================================

predictions = np.zeros(len(y))
fusion_probabilities = np.zeros(len(y))


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
) in enumerate(folds, start=1):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]

    # Normalize using TRAINING fold only
    X_train_scaled, X_test_scaled = (
        standardize_train_test(
            X_train,
            X_test
        )
    )

    weights, bias = train_logistic(
        X_train_scaled,
        y_train,
        l2=L2
    )

    probabilities = sigmoid(
        X_test_scaled @ weights
        + bias
    )

    fold_predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    predictions[test_idx] = (
        fold_predictions
    )

    fusion_probabilities[test_idx] = (
        probabilities
    )

    print(
        f"Fold {fold_number}: "
        f"Train={len(train_idx)}, "
        f"Test={len(test_idx)}"
    )


# ============================================================
# MANUAL METRICS
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


total = TP + TN + FP + FN


accuracy = (
    (TP + TN) / total
)

precision = (
    TP / (TP + FP)
    if (TP + FP) > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)

fpr = (
    FP / (FP + TN)
    if (FP + TN) > 0
    else 0
)

fnr = (
    FN / (FN + TP)
    if (FN + TP) > 0
    else 0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("EXTERNAL ADM FUSION RESULTS")
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