"""
Phase 4 — Train the real meta-fusion model.

This is the trained-model successor to services/final_fusion.py's
rule-based Phase 2 logic. Where Phase 2 used documented, hand-picked
thresholds (because there wasn't enough labelled data to justify a
trained model — see final_fusion.py's docstring), this script fits a
proper multinomial logistic regression (softmax regression) once a
real labelled dataset exists.

Same "no external ML library" philosophy as the existing
train_fusion_model.py — pure NumPy gradient descent, so there's
nothing new to install and the whole model fits in one small,
readable JSON file.

Run AFTER extract_features.py has produced phase4_dataset_features.csv.

Usage:
    python train_final_model.py
"""

import json

import numpy as np

from phase4_common import (
    load_feature_rows,
    split_dataset,
    print_split_summary,
    rows_to_matrix,
    FEATURE_COLUMNS,
    CLASS_NAMES,
    MODEL_DIR,
    FINAL_MODEL_FILE,
)


# ============================================================
# SETTINGS
# ============================================================

L2 = 0.01
LEARNING_RATE = 0.05
ITERATIONS = 6000
SEED = 42


# ============================================================
# SOFTMAX + LOSS
# ============================================================

def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def one_hot(labels, num_classes):
    encoded = np.zeros((len(labels), num_classes), dtype=np.float64)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def fit_softmax_regression(features, labels, num_classes, regularization, iterations, learning_rate):
    """
    Multinomial logistic regression via batch gradient descent.

    features : [N, D] standardized feature matrix
    labels   : [N] integer class indices
    """

    num_samples, num_features = features.shape

    weights = np.zeros((num_features, num_classes), dtype=np.float64)
    bias = np.zeros(num_classes, dtype=np.float64)

    targets = one_hot(labels, num_classes)

    for _ in range(iterations):

        logits = features @ weights + bias

        probabilities = softmax(logits)

        error = probabilities - targets

        grad_weights = (features.T @ error) / num_samples
        grad_weights += regularization * weights / num_samples

        grad_bias = error.mean(axis=0)

        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

    return weights, bias


def predict(features, weights, bias):
    logits = features @ weights + bias
    probabilities = softmax(logits)
    return probabilities.argmax(axis=1), probabilities


def accuracy(predictions, labels):
    return float(np.mean(predictions == labels))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("IMAGEGUARD — PHASE 4 META-FUSION MODEL TRAINING")
    print("=" * 70)

    rows = load_feature_rows()

    print(f"\nTotal labelled images available: {len(rows)}")

    train_rows, val_rows, test_rows = split_dataset(rows, seed=SEED)

    print_split_summary("TRAIN", train_rows)
    print_split_summary("VAL", val_rows)
    print_split_summary("TEST", test_rows)

    if len(train_rows) < 50:
        print(
            "\nWARNING: fewer than 50 training images — this is too small "
            "to train a reliable model. Add more data before trusting "
            "this model's output (see roadmap Phase 4 dataset guidance)."
        )

    X_train, y_train = rows_to_matrix(train_rows)
    X_val, y_val = rows_to_matrix(val_rows)
    X_test, y_test = rows_to_matrix(test_rows)

    # --------------------------------------------------------
    # Standardize using TRAIN statistics only (never fit on val/test —
    # that would leak information about their distribution into
    # training).
    # --------------------------------------------------------

    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    X_train_norm = (X_train - feature_mean) / feature_std
    X_val_norm = (X_val - feature_mean) / feature_std
    X_test_norm = (X_test - feature_mean) / feature_std

    print(f"\nTraining softmax regression ({len(FEATURE_COLUMNS)} features, "
          f"{len(CLASS_NAMES)} classes, {ITERATIONS} iterations)...")

    weights, bias = fit_softmax_regression(
        X_train_norm,
        y_train,
        num_classes=len(CLASS_NAMES),
        regularization=L2,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
    )

    train_preds, _ = predict(X_train_norm, weights, bias)
    val_preds, _ = predict(X_val_norm, weights, bias)
    test_preds, _ = predict(X_test_norm, weights, bias)

    print(f"\nTrain accuracy: {accuracy(train_preds, y_train):.2%}")
    print(f"Val accuracy:   {accuracy(val_preds, y_val):.2%}")
    print(f"Test accuracy:  {accuracy(test_preds, y_test):.2%}")
    print(
        "\n(Train accuracy is NOT the model's real-world accuracy — "
        "run evaluate_final_model.py for the honest per-class "
        "precision/recall/F1/confusion matrix on the held-out test set.)"
    )

    # --------------------------------------------------------
    # Out-of-distribution (OOD) threshold
    # --------------------------------------------------------
    # A trained model can become confidently WRONG when an input's
    # features sit outside anything seen during training — not
    # necessarily one extreme feature, but several moderately unusual
    # features combining. Rather than guessing a cutoff, derive it
    # from the VALIDATION set (data the model didn't train on, but
    # that we know is legitimate/correctly labelled): compute each
    # validation image's overall distance from the training
    # distribution (root-mean-square z-score across all features),
    # then set the threshold at a high percentile of that — i.e.
    # "how far can even a normal, correctly-labelled image
    # legitimately be?". Live images further than this than that are
    # flagged as unreliable regardless of which class the model
    # predicts for them, in services/final_fusion.py.

    val_ood_scores = np.sqrt(np.mean(X_val_norm ** 2, axis=1))

    ood_threshold = float(np.percentile(val_ood_scores, 97.5))

    print(f"\nOut-of-distribution threshold (97.5th percentile of "
          f"validation RMS z-scores): {ood_threshold:.3f}")
    print(
        "Live images whose overall feature profile is farther from "
        "the training distribution than this will be reported as "
        "'Needs Review' regardless of predicted class (see "
        "final_fusion.py)."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = {
        "model_type": "softmax_regression",
        "version": "1.0",

        "class_names": CLASS_NAMES,
        "feature_names": FEATURE_COLUMNS,

        "l2": L2,
        "learning_rate": LEARNING_RATE,
        "iterations": ITERATIONS,
        "seed": SEED,

        "mean": feature_mean.tolist(),
        "std": feature_std.tolist(),

        "weights": weights.tolist(),
        "bias": bias.tolist(),

        "ood_threshold": ood_threshold,

        "training_sample_count": len(train_rows),
        "validation_sample_count": len(val_rows),
        "test_sample_count": len(test_rows),
    }

    with FINAL_MODEL_FILE.open("w", encoding="utf-8") as file:
        json.dump(model, file, indent=4)

    print(f"\nModel saved to: {FINAL_MODEL_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()