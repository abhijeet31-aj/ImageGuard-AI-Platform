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
    k_fold_group_split,
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

# L2 is no longer a hand-picked guess — it's selected by cross-
# validation below (see cross_validate_l2()). These are the
# candidates it chooses from.
L2_CANDIDATES = [0.001, 0.01, 0.05, 0.1, 0.3, 1.0]

LEARNING_RATE = 0.05
ITERATIONS = 6000
SEED = 42
CV_FOLDS = 5


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


def macro_f1_score(predictions, labels, num_classes):
    """
    Macro-averaged F1 — used as the cross-validation selection metric
    instead of plain accuracy, since accuracy alone can look fine
    while quietly ignoring a weak minority class (exactly the
    "Partially AI-Edited" problem this whole exercise is about).
    """

    f1_scores = []

    for class_index in range(num_classes):

        true_positive = int(np.sum((predictions == class_index) & (labels == class_index)))
        false_positive = int(np.sum((predictions == class_index) & (labels != class_index)))
        false_negative = int(np.sum((predictions != class_index) & (labels == class_index)))

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0 else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0 else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def cross_validate_l2(train_rows, candidates, k, num_classes, iterations, learning_rate, seed):
    """
    Proper k-fold, group-aware cross-validation to select L2 — instead
    of guessing a regularization strength, this actually measures
    which one generalizes best on held-out folds of the TRAINING data
    (the val/test sets stay completely untouched by this process).
    """

    folds = k_fold_group_split(train_rows, k=k, seed=seed)

    print(f"\nCross-validating L2 over {len(candidates)} candidate(s), "
          f"{k} folds each (val/test sets are NOT used for this):")

    best_l2 = candidates[0]
    best_score = -1.0

    for l2_candidate in candidates:

        fold_scores = []

        for fold_index in range(k):

            fold_val_rows = folds[fold_index]

            fold_train_rows = [
                row
                for other_index in range(k)
                if other_index != fold_index
                for row in folds[other_index]
            ]

            if not fold_val_rows or not fold_train_rows:
                continue

            X_fold_train, y_fold_train = rows_to_matrix(fold_train_rows)
            X_fold_val, y_fold_val = rows_to_matrix(fold_val_rows)

            fold_mean = X_fold_train.mean(axis=0)
            fold_std = X_fold_train.std(axis=0)
            fold_std[fold_std == 0] = 1.0

            X_fold_train_norm = (X_fold_train - fold_mean) / fold_std
            X_fold_val_norm = (X_fold_val - fold_mean) / fold_std

            fold_weights, fold_bias = fit_softmax_regression(
                X_fold_train_norm,
                y_fold_train,
                num_classes=num_classes,
                regularization=l2_candidate,
                iterations=iterations,
                learning_rate=learning_rate,
            )

            fold_preds, _ = predict(X_fold_val_norm, fold_weights, fold_bias)

            fold_scores.append(
                macro_f1_score(fold_preds, y_fold_val, num_classes)
            )

        mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0

        print(f"  L2={l2_candidate:<8} avg macro-F1 across folds = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_l2 = l2_candidate

    print(f"\nSelected L2={best_l2} (best cross-validated macro-F1: {best_score:.4f})")

    return best_l2


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
    # Cross-validate L2 using ONLY the training rows (val/test stay
    # untouched — this is a proper, data-driven replacement for what
    # used to be a hand-picked L2=0.01 guess).
    # --------------------------------------------------------

    L2 = cross_validate_l2(
        train_rows,
        candidates=L2_CANDIDATES,
        k=CV_FOLDS,
        num_classes=len(CLASS_NAMES),
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )

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
    val_preds, val_probabilities = predict(X_val_norm, weights, bias)
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
    # Needs-Review confidence/margin thresholds — CALIBRATED, not
    # guessed
    # --------------------------------------------------------
    # These used to be fixed constants (0.40 confidence, 0.15 margin)
    # picked by hand in final_fusion.py, with no measurement of how
    # often they'd actually trigger. On real data they ended up
    # flagging a large fraction of ALL predictions as "Needs Review"
    # — not because those images were genuinely ambiguous, but
    # because this model's probability spread is naturally less
    # peaked than the guessed thresholds assumed.
    #
    # Instead, derive them from the VALIDATION set's own behaviour:
    # look at the top-1 probability and margin for predictions that
    # were actually CORRECT, and set the thresholds at a low
    # percentile of that — i.e. "how low does confidence/margin get
    # even when the model is right?". Only predictions below even
    # that get flagged, instead of flagging anything below an
    # arbitrary fixed number.

    val_top2 = -np.sort(-val_probabilities, axis=1)[:, :2]

    val_top1_probability = val_top2[:, 0]
    val_margin = val_top2[:, 0] - val_top2[:, 1]

    val_correct_mask = (val_preds == y_val)

    # Two tiers instead of one binary cutoff — a single hard
    # confident/Needs-Review split forces every moderately-uncertain
    # (but often still correct) prediction into "Needs Review",
    # which is honest but throws away useful information the person
    # could still act on. Instead:
    #   - below the SEVERE percentile  -> "Needs Review" (hard)
    #   - below the MEDIUM percentile (but not severe) -> show the
    #     predicted class with reliability="Medium"
    #   - otherwise -> predicted class with reliability="High"
    MEDIUM_CONFIDENCE_PERCENTILE = 10
    MEDIUM_MARGIN_PERCENTILE = 10

    SEVERE_CONFIDENCE_PERCENTILE = 2
    SEVERE_MARGIN_PERCENTILE = 2

    if val_correct_mask.sum() >= 10:

        needs_review_min_confidence = float(
            np.percentile(val_top1_probability[val_correct_mask], MEDIUM_CONFIDENCE_PERCENTILE)
        )

        needs_review_margin = float(
            np.percentile(val_margin[val_correct_mask], MEDIUM_MARGIN_PERCENTILE)
        )

        severe_min_confidence = float(
            np.percentile(val_top1_probability[val_correct_mask], SEVERE_CONFIDENCE_PERCENTILE)
        )

        severe_margin = float(
            np.percentile(val_margin[val_correct_mask], SEVERE_MARGIN_PERCENTILE)
        )

    else:

        # Not enough correct validation samples to calibrate reliably
        # — fall back to the old fixed defaults rather than computing
        # a meaningless percentile from a handful of points.
        needs_review_min_confidence = 0.40
        needs_review_margin = 0.15
        severe_min_confidence = 0.25
        severe_margin = 0.05

        print(
            "\nWARNING: too few correct validation predictions to "
            "calibrate Needs-Review thresholds — using fallback "
            "defaults."
        )

    would_be_needs_review = (
        (val_top1_probability < severe_min_confidence)
        | (val_margin < severe_margin)
    )

    would_be_medium = (
        ~would_be_needs_review
        & (
            (val_top1_probability < needs_review_min_confidence)
            | (val_margin < needs_review_margin)
        )
    )

    print(f"\nReliability thresholds (calibrated from validation data):")
    print(f"  Severe (-> Needs Review)  : confidence < {severe_min_confidence:.3f} "
          f"or margin < {severe_margin:.3f}")
    print(f"  Medium (-> shown, flagged): confidence < {needs_review_min_confidence:.3f} "
          f"or margin < {needs_review_margin:.3f}")
    print(f"  -> {would_be_needs_review.sum()}/{len(val_preds)} "
          f"({would_be_needs_review.mean():.1%}) would be Needs Review")
    print(f"  -> {would_be_medium.sum()}/{len(val_preds)} "
          f"({would_be_medium.mean():.1%}) would be shown with Medium reliability")
    print(f"  -> {len(val_preds) - would_be_needs_review.sum() - would_be_medium.sum()}/{len(val_preds)} "
          f"would be shown with High reliability")

    # --------------------------------------------------------
    # Cross-check thresholds — ALSO calibrated, not a fixed 0.35
    # --------------------------------------------------------
    # For validation images the model got RIGHT (predicted == true
    # class == "Manipulated" or "AI Generated"), look at how low their
    # own raw signal (manipulationProbability / ai_probability) can
    # legitimately go — this is exactly what the trained model may be
    # correctly picking up on using OTHER features when the raw
    # signal itself is unremarkable. Set the cross-check threshold
    # below even that, so it only fires on a genuine conflict, not on
    # every correct prediction that happens to rely on non-obvious
    # signals.

    CROSS_CHECK_PERCENTILE = 5

    def calibrate_cross_check_threshold(class_name, class_index, raw_value_key):

        correct_class_mask = val_correct_mask & (val_preds == class_index)

        if correct_class_mask.sum() < 10:

            print(
                f"\nWARNING: too few correct validation '{class_name}' "
                f"predictions to calibrate its cross-check threshold — "
                f"using fallback default (0.35)."
            )
            return 0.35

        raw_values = np.array([
            float(row[raw_value_key])
            for row, is_this_class in zip(val_rows, correct_class_mask)
            if is_this_class
        ])

        threshold = float(np.percentile(raw_values, CROSS_CHECK_PERCENTILE))

        print(f"  {class_name} cross-check threshold: {threshold:.3f} "
              f"(was a fixed 0.35 guess before)")

        return threshold

    print(f"\nCross-check thresholds (calibrated from validation data):")

    manipulated_cross_check_threshold = calibrate_cross_check_threshold(
        "Manipulated",
        CLASS_NAMES.index("Manipulated"),
        "manipulationProbability",
    )

    ai_generated_cross_check_threshold = calibrate_cross_check_threshold(
        "AI Generated",
        CLASS_NAMES.index("AI Generated"),
        "ai_probability",
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

    # Two-level OOD: the 97.5th percentile is the MILD boundary (image
    # looks unusual -> still answer, but downgrade reliability), the
    # 99.5th percentile is the SEVERE boundary (image looks like
    # nothing seen in training -> hard "Needs Review"). On the real
    # dataset the hard 97.5th-percentile gate mostly rejected
    # correctly-classified AI-Generated images (small 128-256px images
    # whose forensic feature scale simply differs) — flagged images
    # were MORE accurate than average, the opposite of what a
    # rejection gate should do.
    ood_severe_threshold = float(np.percentile(val_ood_scores, 99.5))

    print(f"\nOut-of-distribution threshold (97.5th percentile of "
          f"validation RMS z-scores): {ood_threshold:.3f}")
    print(f"Severe OOD threshold (99.5th percentile): {ood_severe_threshold:.3f}")
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
        "ood_severe_threshold": ood_severe_threshold,

        "needs_review_min_confidence": needs_review_min_confidence,
        "needs_review_margin": needs_review_margin,

        "severe_min_confidence": severe_min_confidence,
        "severe_margin": severe_margin,

        "manipulated_cross_check_threshold": manipulated_cross_check_threshold,
        "ai_generated_cross_check_threshold": ai_generated_cross_check_threshold,

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