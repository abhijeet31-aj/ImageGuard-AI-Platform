"""
Final Fusion — Phase 2 (rule-based) + Phase 4 (trained model)

Combines two INDEPENDENT signals into one final classification:

    1. fusionAnalysis (existing)   — AI-generation probability,
                                       already a trained logistic
                                       regression over ai_log_odds +
                                       10 forensic features.
    2. manipulationAnalysis (Phase 1) — classical-CV manipulation
                                       probability (noise/ELA/copy-move).

HOW THIS DECIDES WHICH LOGIC TO USE:
    If models/final_fusion_model.json exists (produced by
    train_final_model.py, Phase 4), this module uses that TRAINED
    multinomial logistic regression — the same 12-feature vector
    (ai_log_odds, manipulation_log_odds, 10 forensic features) used
    during training.

    If that file does NOT exist yet, this module falls back to the
    original Phase 2 documented threshold rule. This keeps the app
    working correctly both before and after Phase 4 training — you
    do not need to change app.py or imageController.js either way,
    since combine_final_analysis()'s input/output contract is
    identical regardless of which path runs.

    See train_final_model.py / evaluate_final_model.py for how the
    trained model was fit and evaluated (held-out precision/recall/F1
    per class — check phase4_evaluation_report.csv for the actual
    numbers on YOUR dataset before trusting this in production).
"""

import json
from pathlib import Path

import numpy as np


# ============================================================
# TRAINED MODEL (Phase 4) — loaded once at import time, if present
# ============================================================

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_FINAL_MODEL_FILE = _MODEL_DIR / "final_fusion_model.json"

_TRAINED_MODEL = None

if _FINAL_MODEL_FILE.exists():

    with _FINAL_MODEL_FILE.open("r", encoding="utf-8") as _file:
        _TRAINED_MODEL = json.load(_file)


# Must match phase4_common.FORENSIC_FEATURES order exactly — this is
# duplicated here (rather than imported) so this production module
# doesn't depend on the training-only phase4_common.py being present
# in every deployment.
_FORENSIC_FEATURE_NAMES = [
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

# Even with a trained model, don't force a confident guess when the
# top-2 classes are close, or the top class itself isn't confident —
# same honest philosophy as the Phase 2 rule-based logic, just driven
# by the trained model's own (real, evaluated) probabilities instead
# of hand-picked thresholds.
_NEEDS_REVIEW_MIN_CONFIDENCE = 0.40
_NEEDS_REVIEW_MARGIN = 0.15

_RECOMMENDATIONS = {
    "Authentic":
        "No Edits Detected",

    "AI Generated":
        "Fully AI Generated",

    "Manipulated":
        "Digitally Edited",

    "Partially AI-Edited":
        "AI Edited",
}


def _probability_to_log_odds(probability):
    probability = np.clip(float(probability), 1e-4, 1.0 - 1e-4)
    return float(np.log(probability / (1.0 - probability)))


def _softmax(logits):
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


# ============================================================
# PHASE 4 — TRAINED MODEL PATH
# ============================================================

def _combine_with_trained_model(fusion_analysis, manipulation_analysis):

    # IMPORTANT: extract_features.py trained this model on the RAW
    # ViT ai_probability (ai_detection["ai_probability"]), NOT
    # fusion_probability (which is ai_probability further adjusted by
    # a separate forensic-features logistic regression). Using
    # fusion_probability here would feed the trained model a
    # differently-distributed number than it was trained on, causing
    # systematically unreliable predictions (this was a real bug —
    # if you're reading this after seeing "Needs Review" on almost
    # everything, this line was the cause). Reported as aiLikelihood
    # too, so the displayed number always matches what actually drove
    # the verdict rather than showing a different, possibly
    # contradictory-looking probability.
    ai_probability = float(fusion_analysis.get("ai_probability", 0.0))

    manipulation_probability = float(
        manipulation_analysis.get("manipulationProbability", 0.0)
    )

    forensic = fusion_analysis.get("forensicFeatures", {})

    feature_vector = [
        _probability_to_log_odds(ai_probability),
        _probability_to_log_odds(manipulation_probability),
        *[
            float(forensic.get(name, 0.0))
            for name in _FORENSIC_FEATURE_NAMES
        ],
    ]

    features = np.array(feature_vector, dtype=np.float64)

    mean = np.array(_TRAINED_MODEL["mean"], dtype=np.float64)
    std = np.array(_TRAINED_MODEL["std"], dtype=np.float64)
    weights = np.array(_TRAINED_MODEL["weights"], dtype=np.float64)
    bias = np.array(_TRAINED_MODEL["bias"], dtype=np.float64)
    class_names = _TRAINED_MODEL["class_names"]

    if len(features) != len(mean):
        raise ValueError(
            "Trained model feature count does not match the current "
            "feature vector — was final_fusion_model.json trained "
            "with a different feature set?"
        )

    normalized = (features - mean) / std

    # ----------------------------------------------------
    # OUT-OF-DISTRIBUTION GATE — checked BEFORE trusting any class
    # prediction, and applies regardless of which class the model
    # would otherwise pick.
    # ----------------------------------------------------
    # A trained model can become confidently WRONG when several
    # features sit moderately outside the training distribution at
    # once — no single one extreme enough to look "broken", but
    # collectively enough to tip the decision. ood_threshold (saved
    # by train_final_model.py) is derived from real validation data:
    # the 97.5th-percentile "distance from training distribution" of
    # legitimate, correctly-labelled validation images. If THIS
    # image is farther than that, its feature profile doesn't
    # resemble anything the model was actually trained/validated on
    # — so its classification shouldn't be trusted, whatever it says.

    ood_threshold = _TRAINED_MODEL.get("ood_threshold")

    if ood_threshold is not None:

        ood_score = float(np.sqrt(np.mean(normalized ** 2)))

        if ood_score > ood_threshold:

            return {
                "prediction": "Needs Review",
                "confidence": 0.0,
                "aiLikelihood": round(ai_probability, 4),
                "manipulationLikelihood": round(manipulation_probability, 4),
                "evidence": [
                    f"This image's overall forensic profile is unusually "
                    f"different from the training data (distance score "
                    f"{ood_score:.2f} vs. normal range up to "
                    f"{ood_threshold:.2f}) — the trained model's "
                    f"classification is not reliable enough to trust "
                    f"here, regardless of what it predicted."
                ],
                "recommendation": "Needs Review",
                "modelType": "trained",
            }

    logits = normalized @ weights + bias

    probabilities = _softmax(logits)

    ranked = sorted(
        zip(class_names, probabilities.tolist()),
        key=lambda pair: -pair[1],
    )

    top_class, top_probability = ranked[0]
    second_class, second_probability = ranked[1]

    evidence = [
        f"Trained meta-fusion model: {top_class} "
        f"({top_probability * 100:.1f}%), {second_class} "
        f"({second_probability * 100:.1f}%)."
    ]

    if (
        top_probability < _NEEDS_REVIEW_MIN_CONFIDENCE
        or (top_probability - second_probability) < _NEEDS_REVIEW_MARGIN
    ):

        prediction = "Needs Review"

        confidence = top_probability

        evidence.append(
            "The trained model's top prediction was not confident "
            "enough to commit to a single classification."
        )

        recommendation = "Needs Review"

    else:

        prediction = top_class

        confidence = top_probability

        recommendation = _RECOMMENDATIONS.get(
            prediction,
            "Needs Review"
        )

        if prediction == "Partially AI-Edited":

            evidence.append(
                "Note: this class is historically the hardest for the "
                "trained model to distinguish (see "
                "phase4_evaluation_report.csv) — running Deep Scan is "
                "recommended to confirm with region-level evidence."
            )

        # ----------------------------------------------------
        # Sanity check: does the trained model's "accusation" agree
        # with its own underlying raw signals?
        # ----------------------------------------------------
        # A trained model combining 12 features can become confidently
        # WRONG when several forensic features sit moderately outside
        # the training distribution at once (none individually extreme
        # enough to look like a broken input, but collectively enough
        # to tip the decision) — this happens when the training data
        # doesn't fully represent the range of real-world photos (see
        # final_fusion.py's module docstring / project notes on
        # dataset diversity). Raising the confidence threshold does
        # NOT catch this, since the model IS confident — so instead,
        # cross-check the verdict against the two raw underlying
        # signals it was built from. If the model accuses an image of
        # being Manipulated or AI Generated while BOTH raw signals
        # independently look clean, that disagreement itself is a red
        # flag — downgrade to Needs Review rather than trust the
        # meta-model's combination blindly.

        if prediction == "Manipulated" and manipulation_probability < 0.35:

            prediction = "Needs Review"

            evidence.append(
                f"Conflict: the trained model classified this as "
                f"Manipulated, but the underlying manipulation "
                f"detector itself only scored "
                f"{manipulation_probability * 100:.1f}% (low). "
                f"Downgraded to Needs Review rather than trusting the "
                f"combined model over its own raw signal."
            )

            recommendation = "Needs Review"

        elif prediction == "AI Generated" and ai_probability < 0.35:

            prediction = "Needs Review"

            evidence.append(
                f"Conflict: the trained model classified this as AI "
                f"Generated, but the underlying AI-detection model "
                f"itself only scored {ai_probability * 100:.1f}% "
                f"(low). Downgraded to Needs Review rather than "
                f"trusting the combined model over its own raw signal."
            )

            recommendation = "Needs Review"

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "aiLikelihood": round(ai_probability, 4),
        "manipulationLikelihood": round(manipulation_probability, 4),
        "evidence": evidence,
        "recommendation": recommendation,
        "modelType": "trained",
    }


# ============================================================
# PHASE 2 — RULE-BASED FALLBACK
# ============================================================
# Used automatically when no trained model file exists yet, and as a
# safety fallback if the trained-model path raises for any reason
# (e.g. a stale/incompatible model file).

_AI_HIGH = 0.60
_AI_LOW = 0.40

_MANIPULATION_HIGH = 0.55
_MANIPULATION_LOW = 0.45


def _combine_rule_based(fusion_analysis, manipulation_analysis):
    """
    Decision table (documented, not learned):

        AI signal \\ Manipulation signal   | Low          | High
        -----------------------------------|--------------|------------------
        High (looks AI-generated)          | AI Generated | Needs Review*
        Low  (looks like real camera photo)| Authentic    | Manipulated
        Either signal in its own uncertain band -> Needs Review

        * Both signals firing together is exactly the "partially
          AI-edited" case — Phase 3's tile pipeline (Deep Scan) is
          what actually localizes this.
    """

    ai_probability = float(fusion_analysis.get("fusion_probability", 0.0))

    manipulation_probability = float(
        manipulation_analysis.get("manipulationProbability", 0.0)
    )

    ai_is_high = ai_probability >= _AI_HIGH
    ai_is_low = ai_probability <= _AI_LOW

    manipulation_is_high = manipulation_probability >= _MANIPULATION_HIGH
    manipulation_is_low = manipulation_probability <= _MANIPULATION_LOW

    evidence = []

    if ai_is_high and manipulation_is_low:

        prediction = "AI Generated"

        evidence.append(
            f"AI-generation model indicates a "
            f"{round(ai_probability * 100)}% likelihood of "
            f"AI-generated content."
        )

    elif ai_is_low and manipulation_is_high:

        prediction = "Manipulated"

        evidence.extend(manipulation_analysis.get("evidence", []))

    elif ai_is_low and manipulation_is_low:

        prediction = "Authentic"

    elif ai_is_high and manipulation_is_high:

        prediction = "Needs Review"

        evidence.append(
            "Both AI-generation and manipulation signals were detected "
            "together — this can indicate a partially AI-edited image. "
            "Try running Deep Scan for region-level localization."
        )

    else:

        prediction = "Needs Review"

        evidence.append(
            "AI-generation and/or manipulation signals were inconclusive."
        )

    if not evidence:
        evidence.append("No strong AI-generation or manipulation indicators detected.")

    recommendation = _RECOMMENDATIONS.get(
        prediction,
        "Needs Review"
    )

    ai_certainty = min(abs(ai_probability - 0.5) * 2.0, 1.0)

    manipulation_certainty = min(
        abs(manipulation_probability - 0.5) * 2.2, 1.0
    )

    confidence = round((ai_certainty + manipulation_certainty) / 2.0, 4)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "aiLikelihood": round(ai_probability, 4),
        "manipulationLikelihood": round(manipulation_probability, 4),
        "evidence": evidence,
        "recommendation": recommendation,
        "modelType": "rule_based",
    }


# ============================================================
# PUBLIC ENTRY POINT — unchanged signature since Phase 2
# ============================================================

def combine_final_analysis(fusion_analysis, manipulation_analysis):

    if _TRAINED_MODEL is not None:

        try:

            return _combine_with_trained_model(
                fusion_analysis,
                manipulation_analysis
            )

        except Exception as error:

            print(
                "WARNING: trained final-fusion model failed "
                f"({error}) — falling back to rule-based logic."
            )

    return _combine_rule_based(fusion_analysis, manipulation_analysis)


# ============================================================
# Phase 3 — Deep Scan refinement (unchanged)
# ============================================================

def refine_with_tile_analysis(final_analysis, tile_analysis):
    """
    Upgrade an existing finalAnalysis using the on-demand tile/patch
    scan (Phase 3). This never runs automatically — only when the
    user explicitly requests a "Deep Scan".

    Only changes the verdict when the tile scan gives a clear,
    corroborated signal that the whole-image analysis structurally
    cannot see (localized partial AI editing). Otherwise it leaves
    the original prediction alone and just attaches the tile findings
    as supporting evidence, since — as documented in ai/detector.py —
    individual tile scores are less reliable than the whole-image
    score and shouldn't override it on their own.
    """

    refined = dict(final_analysis)

    evidence = list(final_analysis.get("evidence", []))

    tile_verdict = tile_analysis.get("tileVerdict", "Inconclusive")

    suspicious_regions = tile_analysis.get("suspiciousRegions", [])

    if tile_verdict == "Partially AI-Edited":

        refined["prediction"] = "Partially AI-Edited"

        evidence.append(
            f"Deep Scan found {len(suspicious_regions)} suspicious "
            f"region(s) with high AI-generation scores while the rest "
            f"of the image scored low — consistent with a localized "
            f"AI edit rather than a fully AI-generated or fully "
            f"authentic image."
        )

        refined["recommendation"] = (
            "Localized AI-generated content detected in specific "
            "regions of this image. Review the highlighted areas "
            "before using this image as verified evidence."
        )

    else:

        evidence.append(
            f"Deep Scan (patch-level analysis, {tile_analysis.get('tileCount', 0)} "
            f"regions checked) result: {tile_verdict}. This confirms/"
            f"supplements the whole-image result rather than "
            f"overriding it."
        )

    refined["evidence"] = evidence

    refined["tileAnalysis"] = {
        "tileCount": tile_analysis.get("tileCount", 0),
        "meanScore": tile_analysis.get("meanScore"),
        "stdScore": tile_analysis.get("stdScore"),
        "tileVerdict": tile_verdict,
    }

    refined["suspiciousRegions"] = suspicious_regions

    return refined