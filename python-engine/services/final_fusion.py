"""
Final Fusion — Phase 2

Combines two INDEPENDENT signals into one final classification:

    1. fusionAnalysis (existing)   — AI-generation probability,
                                       already a trained logistic
                                       regression over ai_log_odds +
                                       10 forensic features.
    2. manipulationAnalysis (Phase 1) — classical-CV manipulation
                                       probability (noise/ELA/copy-move).

IMPORTANT — why this is rule-based, not a trained model:
    A proper stacking/meta-fusion classifier needs a labelled dataset
    to fit on. As of Phase 2, the only labelled manipulated/authentic
    examples available are the 10 sample images already in
    test_images/ (5 original, 5 edited) — nowhere near enough to
    train a model without it just memorizing those 10 images.
    Training "a model" on 10 examples and reporting it as a real
    classifier would be a fabricated accuracy claim, which this
    project explicitly avoids.

    So Phase 2 uses a transparent, documented threshold rule instead.
    This is honest about being a heuristic, and it directly sets up
    Phase 4: once a real labelled dataset (CASIA/RAISE/synthesized
    partial edits) is collected, this function's logic can be
    replaced by a properly trained + evaluated meta-model without
    changing its input/output contract.
"""


# Margins around each detector's own decision threshold. The AI
# fusion model is a trained classifier (model threshold = 0.5), so it
# gets a narrower "uncertain" band. The manipulation detector is a
# Phase-1 classical-CV heuristic with more inherent uncertainty, so
# it keeps the same wider band already used in manipulation.py.
AI_HIGH = 0.60
AI_LOW = 0.40

MANIPULATION_HIGH = 0.55
MANIPULATION_LOW = 0.45


def combine_final_analysis(fusion_analysis, manipulation_analysis):
    """
    Combine aiDetection/fusionAnalysis with manipulationAnalysis into
    a single finalAnalysis result covering all 4 classes.

    Decision table (documented, not learned):

        AI signal \\ Manipulation signal   | Low          | High
        -----------------------------------|--------------|------------------
        High (looks AI-generated)          | AI Generated | Needs Review*
        Low  (looks like real camera photo)| Authentic    | Manipulated
        Either signal in its own uncertain band → Needs Review

        * Both signals firing together is exactly the "partially
          AI-edited" case described in the roadmap — Phase 3's tile
          pipeline is what actually localizes this. Until then it is
          honestly reported as Needs Review with that evidence noted,
          not force-labelled.
    """

    ai_probability = float(fusion_analysis.get("fusion_probability", 0.0))

    manipulation_probability = float(
        manipulation_analysis.get("manipulationProbability", 0.0)
    )

    ai_is_high = ai_probability >= AI_HIGH
    ai_is_low = ai_probability <= AI_LOW

    manipulation_is_high = manipulation_probability >= MANIPULATION_HIGH
    manipulation_is_low = manipulation_probability <= MANIPULATION_LOW

    evidence = []

    # --------------------------------------------------------
    # Decision logic
    # --------------------------------------------------------

    if ai_is_high and manipulation_is_low:

        prediction = "AI Generated"

        evidence.append(
            f"AI-generation model indicates a "
            f"{round(ai_probability * 100)}% likelihood of "
            f"AI-generated content."
        )

        recommendation = (
            "This image contains likely AI-generated content. "
            "Do not use this image as verified evidence without "
            "independent confirmation."
        )

    elif ai_is_low and manipulation_is_high:

        prediction = "Manipulated"

        recommendation = (
            "Possible image manipulation detected. Review suspicious "
            "regions before using this image as verified evidence."
        )

        evidence.extend(manipulation_analysis.get("evidence", []))

    elif ai_is_low and manipulation_is_low:

        prediction = "Authentic"

        recommendation = (
            "No significant AI or manipulation signals were detected."
        )

    elif ai_is_high and manipulation_is_high:

        # Both specialists are firing — consistent with a partially
        # AI-edited image, but Phase 2 cannot localize this (that is
        # Phase 3's tile pipeline). Report honestly instead of
        # guessing which one is "more right".
        prediction = "Needs Review"

        evidence.append(
            "Both AI-generation and manipulation signals were detected "
            "together — this can indicate a partially AI-edited image. "
            "Region-level localization is not yet available (planned "
            "for a future update)."
        )

        recommendation = (
            "Multiple conflicting signals detected. Manual review is "
            "recommended before trusting this image."
        )

    else:

        # At least one signal is in its own uncertain middle band —
        # don't force a confident guess out of an unclear reading.
        prediction = "Needs Review"

        evidence.append(
            "AI-generation and/or manipulation signals were inconclusive."
        )

        recommendation = (
            "Image requires further verification before being "
            "considered trustworthy."
        )

    if not evidence:
        evidence.append("No strong AI-generation or manipulation indicators detected.")

    # --------------------------------------------------------
    # Confidence — how far both signals sit from their own
    # uncertain middle band, averaged. Not a calibrated
    # probability; see module docstring.
    # --------------------------------------------------------

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
    }