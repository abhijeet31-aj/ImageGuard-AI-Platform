import csv
import json
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AI_FILE = BASE_DIR / "ai_validation_predictions.csv"
FORENSIC_FILE = BASE_DIR / "forensic_validation_results.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "fusion_model.json"


# ============================================================
# SETTINGS
# ============================================================

L2 = 0.01
LEARNING_RATE = 0.05
ITERATIONS = 4000
THRESHOLD = 0.50


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
# LOAD CSV
# ============================================================

def load_csv(path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


prediction_rows = load_csv(AI_FILE)
forensic_rows = load_csv(FORENSIC_FILE)


# ============================================================
# BUILD LOOKUPS
# ============================================================

prediction_lookup = {
    (row["filename"], row["true_label"]): row
    for row in prediction_rows
}

forensic_lookup = {
    (row["filename"], row["label"]): row
    for row in forensic_rows
}


if set(prediction_lookup) != set(forensic_lookup):
    raise ValueError(
        "Prediction and forensic CSVs do not contain the same keyed rows."
    )


# ============================================================
# BUILD DATASET
# ============================================================

keys = sorted(forensic_lookup)

labels = np.array(
    [1 if label == "AI" else 0 for _, label in keys],
    dtype=np.float64,
)

ai_probabilities = np.array(
    [
        float(prediction_lookup[key]["ai_probability"])
        for key in keys
    ],
    dtype=np.float64,
)


# ============================================================
# AI PROBABILITY → LOG ODDS
# ============================================================

clipped_probability = np.clip(
    ai_probabilities,
    1e-4,
    1 - 1e-4,
)

detector_log_odds = np.log(
    clipped_probability /
    (1 - clipped_probability)
)


# ============================================================
# FORENSIC FEATURES
# ============================================================

forensic_matrix = np.array(
    [
        [
            float(forensic_lookup[key][feature])
            for feature in FORENSIC_FEATURES
        ]
        for key in keys
    ],
    dtype=np.float64,
)


# ============================================================
# FINAL FUSION MATRIX
# ============================================================

fusion_matrix = np.column_stack(
    [
        detector_log_odds,
        forensic_matrix,
    ]
)


# ============================================================
# STANDARDIZATION
# ============================================================

feature_mean = fusion_matrix.mean(axis=0)
feature_std = fusion_matrix.std(axis=0)

feature_std[feature_std == 0] = 1.0

normalized_features = (
    fusion_matrix - feature_mean
) / feature_std


# ============================================================
# SIGMOID
# ============================================================

def sigmoid(values):
    values = np.clip(values, -500, 500)

    return 1.0 / (
        1.0 + np.exp(-values)
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def fit_logistic_regression(
    features,
    labels,
    regularization,
    iterations=4000,
):
    weights = np.zeros(
        features.shape[1] + 1,
        dtype=np.float64,
    )

    design = np.column_stack(
        [
            np.ones(len(features)),
            features,
        ]
    )

    penalty = np.r_[
        0.0,
        np.full(
            features.shape[1],
            regularization,
        ),
    ]

    for _ in range(iterations):

        probabilities = sigmoid(
            design @ weights
        )

        gradient = (
            design.T @
            (probabilities - labels)
        ) / len(labels)

        gradient += (
            penalty *
            weights /
            len(labels)
        )

        weights -= (
            LEARNING_RATE *
            gradient
        )

    return weights


# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("IMAGEGUARD - PRODUCTION FUSION MODEL TRAINING")
print("=" * 70)

print()
print("Samples:", len(labels))
print("REAL:", int(np.sum(labels == 0)))
print("AI:", int(np.sum(labels == 1)))

weights = fit_logistic_regression(
    normalized_features,
    labels,
    regularization=L2,
    iterations=ITERATIONS,
)


# ============================================================
# TRAINING CHECK
# ============================================================

design = np.column_stack(
    [
        np.ones(len(normalized_features)),
        normalized_features,
    ]
)

probabilities = sigmoid(
    design @ weights
)

predictions = probabilities >= THRESHOLD

accuracy = np.mean(
    predictions == labels
)


print()
print("L2:", L2)
print("Learning rate:", LEARNING_RATE)
print("Iterations:", ITERATIONS)
print("Threshold:", THRESHOLD)
print(f"Training accuracy: {accuracy:.2%}")


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


model = {
    "model_type": "logistic_regression",
    "version": "1.0",

    "threshold": THRESHOLD,

    "l2": L2,
    "learning_rate": LEARNING_RATE,
    "iterations": ITERATIONS,

    "features": [
        "ai_log_odds",
        *FORENSIC_FEATURES,
    ],

    "mean": feature_mean.tolist(),

    "std": feature_std.tolist(),

    "weights": weights[1:].tolist(),

    "bias": float(weights[0]),
}


with MODEL_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        model,
        file,
        indent=4,
    )


print()
print("Model saved successfully:")
print(MODEL_FILE)
print("=" * 70)