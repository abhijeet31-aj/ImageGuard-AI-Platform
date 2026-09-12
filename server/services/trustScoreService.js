const clamp = (value, min = 0, max = 100) => {
    return Math.max(min, Math.min(max, value));
};


export const calculateTrustScore = ({
    fusionAnalysis,
    manipulationAnalysis,
    metadataAnalysis,
    finalAnalysis
}) => {

    // --------------------------------------------------
    // 1. AI-GENERATION SIGNAL — 50 POINTS
    // --------------------------------------------------
    // FIX (Phase 2): previously this used the raw
    // aiDetection.ai_probability directly, ignoring fusionAnalysis
    // entirely — so the trustScore saved to the database could
    // disagree with the fusionAnalysis verdict shown on the Result
    // page. fusion_probability already incorporates the forensic
    // features on top of the raw AI probability, so it is used here.

    const aiProbability = clamp(
        Number(fusionAnalysis?.fusion_probability ?? 0),
        0,
        1
    );

    const aiScore = (1 - aiProbability) * 50;


    // --------------------------------------------------
    // 2. MANIPULATION SIGNAL — 25 POINTS
    // --------------------------------------------------
    // FIX (Phase 2): this signal did not exist before — manipulation
    // detection (Phase 1) was not part of the trust score at all.

    const manipulationProbability = clamp(
        Number(manipulationAnalysis?.manipulationProbability ?? 0),
        0,
        1
    );

    const manipulationScore = (1 - manipulationProbability) * 25;


    // --------------------------------------------------
    // 3. METADATA — 15 POINTS
    // --------------------------------------------------
    // Unchanged from before.

    const integrityScore = clamp(
        Number(metadataAnalysis?.integrityScore ?? 0),
        0,
        100
    );

    const metadataScore =
        (integrityScore / 100) * 15;


    // --------------------------------------------------
    // 4. CORROBORATION BONUS — 10 POINTS
    // --------------------------------------------------
    // FIX (Phase 2): previously ELA (10) and forgery (15) were
    // hardcoded fixed points regardless of their actual values, so
    // they had zero real effect on the score. They are now folded
    // into the AI/manipulation signals above (which already consume
    // the ELA/forgery features via fusion_probability and
    // manipulationAnalysis), plus a small explicit bonus when both
    // independent detectors agree the image is clean — agreement
    // between independent signals is itself meaningful evidence.

    let corroborationScore = 0;

    if (aiProbability <= 0.35 && manipulationProbability <= 0.35) {

        corroborationScore = 10;
    }


    // --------------------------------------------------
    // FINAL TRUST SCORE
    // --------------------------------------------------

    let trustScore = Math.round(
        clamp(
            aiScore +
            manipulationScore +
            metadataScore +
            corroborationScore
        )
    );


    // --------------------------------------------------
    // NEEDS REVIEW CAP
    // --------------------------------------------------
    // If the Python engine's finalAnalysis already flagged this as
    // "Needs Review" (signals disagree or sit in an uncertain band),
    // don't let the trust score independently claim high confidence
    // in either direction — cap it into the Medium band so the
    // number shown matches the honesty of the verdict.

    if (finalAnalysis?.prediction === "Needs Review") {

        trustScore = clamp(trustScore, 40, 65);
    }


    // --------------------------------------------------
    // RISK LEVEL
    // --------------------------------------------------

    let riskLevel;

    if (trustScore >= 70) {

        riskLevel = "Low";

    } else if (trustScore >= 40) {

        riskLevel = "Medium";

    } else {

        riskLevel = "High";
    }


    // --------------------------------------------------
    // RECOMMENDATION
    // --------------------------------------------------
    // Prefer the Python engine's own finalAnalysis recommendation
    // when available, since it already reflects the specific
    // combination of signals that fired. Fall back to a generic
    // score-based recommendation otherwise.

    let recommendation = finalAnalysis?.recommendation;

    if (!recommendation) {

        if (trustScore >= 70) {

            recommendation =
                "Image appears trustworthy based on the available evidence.";

        } else if (trustScore >= 40) {

            recommendation =
                "Image requires further verification before being considered trustworthy.";

        } else {

            recommendation =
                "Image shows significant authenticity risk and requires verification.";
        }
    }


    // --------------------------------------------------
    // RESULT
    // --------------------------------------------------

    return {

        trustScore,

        riskLevel,

        recommendation,

        scoreBreakdown: {

            ai: Number(
                aiScore.toFixed(2)
            ),

            manipulation: Number(
                manipulationScore.toFixed(2)
            ),

            metadata: Number(
                metadataScore.toFixed(2)
            ),

            corroboration: corroborationScore

        }

    };
};