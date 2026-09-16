"""
Phase 4 — Honest evaluation of the trained meta-fusion model.

Reports precision/recall/F1/confusion matrix on the HELD-OUT TEST
SPLIT ONLY — never on the training data. Per the project's own
standing rule: training accuracy is never reported as the model's
real accuracy.

Usage:
    python evaluate_final_model.py
    (run AFTER train_final_model.py)
"""

import csv
import json

import numpy as np

from phase4_common import (
    load_feature_rows,
    split_dataset,
    print_split_summary,
    rows_to_matrix,
    CLASS_NAMES,
    FINAL_MODEL_FILE,
    EVALUATION_REPORT_FILE,
)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_model():
    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{FINAL_MODEL_FILE} not found — run train_final_model.py first."
        )

    with FINAL_MODEL_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1

    return matrix


def per_class_metrics(matrix, class_names):
    """
    matrix[i, j] = number of samples whose TRUE class is i and
    PREDICTED class is j. Rows = truth, columns = prediction.
    """

    metrics = []

    for index, class_name in enumerate(class_names):

        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        support = matrix[index, :].sum()

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics.append({
            "class": class_name,
            "support": int(support),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    return metrics


def main():

    print("=" * 70)
    print("IMAGEGUARD — PHASE 4 MODEL EVALUATION (held-out test set)")
    print("=" * 70)

    model = load_model()

    class_names = model["class_names"]

    mean = np.array(model["mean"])
    std = np.array(model["std"])
    weights = np.array(model["weights"])
    bias = np.array(model["bias"])
    ood_threshold = model.get("ood_threshold")

    rows = load_feature_rows()

    # Re-derive the exact same split used during training (same seed,
    # same deterministic grouping logic) rather than storing it
    # separately — keeps train/evaluate always in sync.
    _train_rows, _val_rows, test_rows = split_dataset(
        rows,
        seed=model["seed"]
    )

    print_split_summary("TEST (held out, never seen during training)", test_rows)

    if not test_rows:
        print("\nNo test rows available — cannot evaluate.")
        return

    X_test, y_test = rows_to_matrix(test_rows)

    X_test_norm = (X_test - mean) / std

    logits = X_test_norm @ weights + bias

    probabilities = softmax(logits)

    predictions = probabilities.argmax(axis=1)

    matrix = confusion_matrix(y_test, predictions, len(class_names))

    metrics = per_class_metrics(matrix, class_names)

    overall_accuracy = float(np.mean(predictions == y_test))

    macro_precision = float(np.mean([m["precision"] for m in metrics]))
    macro_recall = float(np.mean([m["recall"] for m in metrics]))
    macro_f1 = float(np.mean([m["f1"] for m in metrics]))

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(f"\nTest accuracy: {overall_accuracy:.2%}")

    print("\nPer-class metrics:")
    print(f"{'Class':<22s}{'Support':>10s}{'Precision':>12s}{'Recall':>10s}{'F1':>8s}")

    for m in metrics:
        print(
            f"{m['class']:<22s}{m['support']:>10d}"
            f"{m['precision']:>12.2%}{m['recall']:>10.2%}{m['f1']:>8.2%}"
        )

    print(f"\nMacro avg — Precision: {macro_precision:.2%}  "
          f"Recall: {macro_recall:.2%}  F1: {macro_f1:.2%}")

    # --------------------------------------------------------
    # Out-of-distribution gate impact on the TEST set
    # --------------------------------------------------------
    # Shows what fraction of held-out images would be routed to
    # "Needs Review" by the OOD gate (see final_fusion.py), and how
    # much cleaner accuracy is on the remaining "confidently
    # in-distribution" subset. If accuracy on the OOD-flagged subset
    # is much worse than the rest, that confirms the gate is doing
    # its job — catching the cases the model is unreliable on.

    if ood_threshold is not None:

        test_ood_scores = np.sqrt(np.mean(X_test_norm ** 2, axis=1))

        is_ood = test_ood_scores > ood_threshold

        print(f"\nOut-of-distribution gate (threshold={ood_threshold:.3f}):")
        print(f"  Flagged as Needs Review (OOD): {is_ood.sum()} / {len(is_ood)} "
              f"({is_ood.mean():.1%})")

        if (~is_ood).sum() > 0:
            in_dist_accuracy = float(np.mean(predictions[~is_ood] == y_test[~is_ood]))
            print(f"  Accuracy on the remaining (in-distribution) images: {in_dist_accuracy:.2%}")

        if is_ood.sum() > 0:
            ood_accuracy = float(np.mean(predictions[is_ood] == y_test[is_ood]))
            print(f"  Accuracy on the flagged (out-of-distribution) images: {ood_accuracy:.2%}")
            print("  (low accuracy here is EXPECTED and is exactly why these get flagged)")

    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    header = " " * 24 + "".join(f"{name[:12]:>14s}" for name in class_names)
    print(header)

    for index, class_name in enumerate(class_names):
        row_str = "".join(f"{value:>14d}" for value in matrix[index])
        print(f"{class_name:<24s}{row_str}")

    # --------------------------------------------------------
    # Save report to CSV
    # --------------------------------------------------------

    with EVALUATION_REPORT_FILE.open("w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(["class", "support", "precision", "recall", "f1"])

        for m in metrics:
            writer.writerow([
                m["class"], m["support"],
                m["precision"], m["recall"], m["f1"]
            ])

        writer.writerow([])
        writer.writerow(["overall_accuracy", round(overall_accuracy, 4)])
        writer.writerow(["macro_precision", round(macro_precision, 4)])
        writer.writerow(["macro_recall", round(macro_recall, 4)])
        writer.writerow(["macro_f1", round(macro_f1, 4)])

        writer.writerow([])
        writer.writerow(["confusion_matrix (rows=true, cols=predicted)"])
        writer.writerow([""] + class_names)

        for index, class_name in enumerate(class_names):
            writer.writerow([class_name] + list(matrix[index]))

    print(f"\nFull report saved to: {EVALUATION_REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()