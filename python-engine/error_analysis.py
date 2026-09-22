"""
Phase 4 — Error analysis over the held-out test set.

Unlike diagnose_prediction.py (which looks at ONE image at a time),
this script looks at ALL misclassifications together to find
SYSTEMATIC patterns — e.g. "when the model wrongly predicts Partially
AI-Edited for an Authentic image, feature X is consistently elevated
compared to correctly-classified Authentic images". That's the kind
of finding that tells you what to actually fix (more data covering
that feature range, or reconsidering that feature), rather than
patching one photo at a time.

Usage:
    python error_analysis.py
    (run AFTER train_final_model.py)
"""

import json
from collections import defaultdict

import numpy as np

from phase4_common import (
    load_feature_rows,
    split_dataset,
    rows_to_matrix,
    FEATURE_COLUMNS,
    CLASS_NAMES,
    FINAL_MODEL_FILE,
)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def main():

    print("=" * 78)
    print("IMAGEGUARD — PHASE 4 ERROR ANALYSIS (held-out test set)")
    print("=" * 78)

    with FINAL_MODEL_FILE.open(encoding="utf-8") as file:
        model = json.load(file)

    mean = np.array(model["mean"])
    std = np.array(model["std"])
    weights = np.array(model["weights"])
    bias = np.array(model["bias"])
    class_names = model["class_names"]

    rows = load_feature_rows()

    _train_rows, _val_rows, test_rows = split_dataset(rows, seed=model["seed"])

    if not test_rows:
        print("No test rows available.")
        return

    X_test, y_test = rows_to_matrix(test_rows)

    z_scores = (X_test - mean) / std

    logits = z_scores @ weights + bias
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)

    is_wrong = predictions != y_test

    print(f"\nTotal test images: {len(test_rows)}")
    print(f"Misclassified:     {is_wrong.sum()} ({is_wrong.mean():.1%})")

    # --------------------------------------------------------
    # 1. Confusion pair breakdown — which mistakes happen most
    # --------------------------------------------------------

    confusion_pairs = defaultdict(int)

    for true_idx, pred_idx in zip(y_test[is_wrong], predictions[is_wrong]):
        confusion_pairs[(class_names[true_idx], class_names[pred_idx])] += 1

    print("\nMost common mistakes (true -> predicted):")

    for (true_name, pred_name), count in sorted(
        confusion_pairs.items(), key=lambda item: -item[1]
    ):
        print(f"  {true_name:22s} -> {pred_name:22s}  {count} image(s)")

    # --------------------------------------------------------
    # 2. For the TOP FEW most common confusion pairs, find which
    #    features differ most between the correctly-classified
    #    group and the wrongly-classified group of the same true
    #    class. This pinpoints what's actually misleading the model.
    # --------------------------------------------------------

    if not confusion_pairs:
        print("\nNo misclassifications — nothing to analyze further.")
        return

    top_confusion_pairs = sorted(
        confusion_pairs.items(), key=lambda item: -item[1]
    )[:3]

    for (worst_true_name, worst_pred_name), worst_count in top_confusion_pairs:

        print(f"\n{'-' * 78}")
        print(
            f"Deep dive: {worst_true_name} images wrongly predicted as "
            f"{worst_pred_name} ({worst_count} case(s))"
        )
        print("-" * 78)

        true_class_index = class_names.index(worst_true_name)

        same_true_class_mask = y_test == true_class_index

        correct_mask = same_true_class_mask & (~is_wrong)
        wrong_mask = same_true_class_mask & is_wrong & (predictions == class_names.index(worst_pred_name))

        if correct_mask.sum() == 0 or wrong_mask.sum() == 0:
            print("Not enough samples in one of the groups to compare.")
            continue

        correct_features = z_scores[correct_mask]
        wrong_features = z_scores[wrong_mask]

        print(
            f"\nComparing {wrong_mask.sum()} wrongly-classified vs "
            f"{correct_mask.sum()} correctly-classified '{worst_true_name}' images "
            f"(values are z-scores relative to training distribution):\n"
        )

        print(f"{'Feature':<24s}{'correct (mean)':>16s}{'wrong (mean)':>16s}{'difference':>14s}")

        differences = []

        for index, feature_name in enumerate(FEATURE_COLUMNS):

            correct_mean = float(correct_features[:, index].mean())
            wrong_mean = float(wrong_features[:, index].mean())
            diff = wrong_mean - correct_mean

            differences.append((feature_name, diff))

            flag = "  <-- LARGE DIFFERENCE" if abs(diff) > 0.5 else ""

            print(f"{feature_name:<24s}{correct_mean:>16.3f}{wrong_mean:>16.3f}{diff:>14.3f}{flag}")

        differences.sort(key=lambda item: -abs(item[1]))

        print(f"\nMost responsible feature(s) for this confusion:")

        for feature_name, diff in differences[:3]:
            direction = "higher" if diff > 0 else "lower"
            print(
                f"  - {feature_name}: {direction} in misclassified images "
                f"(diff={diff:+.3f} std deviations)"
            )

    print(
        "\nWhat this means: these features push the model toward the "
        "wrong class specifically when they sit at these levels. To "
        "fix this properly:\n"
        "  1. Collect more training examples of the TRUE class that "
        "specifically have these feature values (closes the gap in "
        "the training data)\n"
        "  2. Or reconsider whether the flagged feature(s) reliably "
        "separate these classes at all — if not, it may be adding "
        "more noise than signal\n"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()