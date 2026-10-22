const clamp = (value, min = 0, max = 100) => {
    return Math.max(min, Math.min(max, value));
};


// ------------------------------------------------------------------
// Trust Score = calibrated-style probability that the image is
// Authentic (0-100), taken from the SAME final model result that
// produces the verdict. This keeps the verdict and the Trust Score
// consistent on the Result page (single source of truth).
//
// NOTE: the class probabilities are the trained model's softmax
// output; they are NOT yet calibrated on validation data (planned).
// Records produced before this change, or by the rule-based fallback
// (no classProbabilities), use the legacy multi-signal formula below.
// ------------------------------------------------------------------

const legacyTrustScore = ({
    fusionAnalysis,
    manipulationAnalysis,
    metadataAnalysis,
    finalAnalysis
}) => {

    const aiProbability = clamp(
        Number(fusionAnalysis?.fusion_probability ?? 0),
        0,
        1
    );

    const aiScore = (1 - aiProbability) * 50;

    const manipulationProbability = clamp(
        Number(manipulationAnalysis?.manipulationProbability ?? 0),
        0,
        1
    );

    const manipulationScore = (1 - manipulationProbability) * 25;

    const integrityScore = clamp(
        Number(metadataAnalysis?.integrityScore ?? 0),
        0,
        100
    );

    const metadataScore = (integrityScore / 100) * 15;

    let corroborationScore = 0;

    if (aiProbability <= 0.35 && manipulationProbability <= 0.35) {

        corroborationScore = 10;
    }

    let trustScore = Math.round(
        clamp(
            aiScore +
            manipulationScore +
            metadataScore +
            corroborationScore
        )
    );

    if (finalAnalysis?.prediction === "Needs Review") {

        trustScore = clamp(trustScore, 40, 65);

    } else if (finalAnalysis?.reliability === "Medium") {

        trustScore = clamp(trustScore - 10, 0, 80);
    }

    return {
        trustScore,
        scoreBreakdown: {
            ai: Number(aiScore.toFixed(2)),
            manipulation: Number(manipulationScore.toFixed(2)),
            metadata: Number(metadataScore.toFixed(2)),
            corroboration: corroborationScore
        }
    };
};


export const calculateTrustScore = ({
    fusionAnalysis,
    manipulationAnalysis,
    metadataAnalysis,
    finalAnalysis
}) => {

    const authenticProbability =
        finalAnalysis?.classProbabilities?.["Authentic"];

    let trustScore;
    let scoreBreakdown;

    if (
        typeof authenticProbability === "number" &&
        Number.isFinite(authenticProbability)
    ) {

        // New: Trust Score = P(Authentic) x 100 from the final model.
        trustScore = Math.round(
            clamp(authenticProbability * 100)
        );

        // "Needs Review" means the evidence is inconclusive, so the
        // score must not claim certainty in either direction. Keep it
        // inside the Medium band (existing behaviour, unchanged).
        if (finalAnalysis?.prediction === "Needs Review") {

            trustScore = clamp(trustScore, 40, 65);
        }

        scoreBreakdown = {
            basis: "finalAnalysis.classProbabilities",
            authenticProbability: Number(
                (authenticProbability * 100).toFixed(1)
            )
        };

    } else {

        ({ trustScore, scoreBreakdown } = legacyTrustScore({
            fusionAnalysis,
            manipulationAnalysis,
            metadataAnalysis,
            finalAnalysis
        }));
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

        scoreBreakdown

    };
};