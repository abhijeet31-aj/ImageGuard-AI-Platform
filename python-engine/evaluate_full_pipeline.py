"""
Phase 4 — Evaluate the FULL production pipeline (not just the raw
softmax model).

evaluate_final_model.py measures the trained model's raw accuracy.
But the live app doesn't use the raw model directly — every
prediction also passes through services/final_fusion.py's safety
layers: the OOD gate, the confidence/margin "Needs Review" check, the
Manipulated/AI-Generated cross-check, and the stricter Partially
AI-Edited threshold. Those were each added to fix ONE specific
reported problem, one at a time — their COMBINED effect on real
held-out data has never actually been measured until now.

This script calls the real combine_final_analysis() (the exact same
function app.py uses) for every held-out test image, and reports:
  - what fraction end up as "Needs Review" overall
  - accuracy on the remaining "confident" subset
  - which safety layer is responsible, per Needs Review case

Usage:
    python evaluate_full_pipeline.py
    (run AFTER train_final_model.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "services"))

from phase4_common import (
    load_feature_rows,
    split_dataset,
    FORENSIC_FEATURES,
    CLASS_NAMES,
)

from services.final_fusion import combine_final_analysis


def build_fusion_and_manipulation_analysis(row):

    forensic = {
        name: float(row[name])
        for name in FORENSIC_FEATURES
    }

    fusion_analysis = {
        "ai_probability": float(row["ai_probability"]),
        "fusion_probability": float(row["ai_probability"]),  # not used by the trained-model path
        "forensicFeatures": forensic,
        "ai_regional_std": float(row["aiRegionalStd"]),
    }

    manipulation_analysis = {
        "manipulationProbability": float(row["manipulationProbability"]),
        "evidence": [],
    }

    return fusion_analysis, manipulation_analysis


def classify_needs_review_reason(evidence_list):

    joined = " ".join(evidence_list)

    if "unusually different from the training data" in joined:
        return "OOD gate"

    if "Conflict:" in joined:
        return "Cross-check conflict"

    if "not confident enough" in joined:
        return "Low confidence / narrow margin"

    if "inconclusive" in joined:
        return "Rule-based fallback (no trained model?)"

    return "Other"


def main():

    print("=" * 78)
    print("IMAGEGUARD — FULL PIPELINE EVALUATION (all safety layers, held-out test set)")
    print("=" * 78)

    rows = load_feature_rows()

    # Need the model's seed to reproduce the exact same split used
    # during training — read it straight from the saved model file.
    import json
    from phase4_common import FINAL_MODEL_FILE

    with FINAL_MODEL_FILE.open(encoding="utf-8") as file:
        model = json.load(file)

    _train_rows, _val_rows, test_rows = split_dataset(rows, seed=model["seed"])

    print(f"\nTest set: {len(test_rows)} images\n")

    total = len(test_rows)
    needs_review_count = 0
    correct_among_confident = 0
    confident_count = 0

    reason_counts = {}

    reason_by_true_class = {}

    for row in test_rows:

        fusion_analysis, manipulation_analysis = build_fusion_and_manipulation_analysis(row)

        result = combine_final_analysis(fusion_analysis, manipulation_analysis)

        true_class = row["class_label"]

        if result["prediction"] == "Needs Review":

            needs_review_count += 1

            reason = classify_needs_review_reason(result.get("evidence", []))

            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            reason_by_true_class.setdefault(true_class, {})
            reason_by_true_class[true_class][reason] = (
                reason_by_true_class[true_class].get(reason, 0) + 1
            )

        else:

            confident_count += 1

            if result["prediction"] == true_class:
                correct_among_confident += 1

    print(f"Needs Review triggered: {needs_review_count} / {total} "
          f"({needs_review_count / total:.1%})")

    print(f"Confident predictions:  {confident_count} / {total} "
          f"({confident_count / total:.1%})")

    if confident_count > 0:
        print(f"Accuracy on confident predictions: "
              f"{correct_among_confident / confident_count:.2%}")

    print("\nWhy Needs Review triggered (by safety layer):")

    for reason, count in sorted(reason_counts.items(), key=lambda item: -item[1]):
        print(f"  {reason:<35s} {count:>4d}  ({count / total:.1%} of all test images)")

    print("\nNeeds Review rate BY TRUE CLASS (which classes get over-flagged):")

    for class_name in CLASS_NAMES:

        class_total = sum(1 for row in test_rows if row["class_label"] == class_name)

        class_needs_review = sum(
            reason_by_true_class.get(class_name, {}).values()
        )

        if class_total > 0:
            print(f"  {class_name:<22s} {class_needs_review}/{class_total} "
                  f"({class_needs_review / class_total:.1%}) flagged Needs Review")

    print("\n" + "=" * 78)
    print(
        "If 'Low confidence / narrow margin' dominates, the general "
        "margin/confidence thresholds are too strict for this model's "
        "actual probability spread — they were guessed, not measured, "
        "and should be relaxed based on the numbers above."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()