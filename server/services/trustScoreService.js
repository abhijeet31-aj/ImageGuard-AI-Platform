const clamp = (value, min = 0, max = 100) => {
    return Math.max(min, Math.min(max, value));
};


export const calculateTrustScore = ({
    aiDetection,
    metadataAnalysis
}) => {

    // --------------------------------------------------
    // 1. AI DETECTION — 60 POINTS
    // --------------------------------------------------

    const aiProbability = clamp(
        Number(aiDetection?.ai_probability || 0),
        0,
        1
    );

    const aiScore =
        (1 - aiProbability) * 60;


    // --------------------------------------------------
    // 2. METADATA — 15 POINTS
    // --------------------------------------------------

    const integrityScore = clamp(
        Number(
            metadataAnalysis?.integrityScore || 0
        )
    );

    const metadataScore =
        (integrityScore / 100) * 15;


    // --------------------------------------------------
    // 3. ELA — 10 POINTS
    // --------------------------------------------------
    // ELA is currently treated as supporting evidence.
    // We do not use an arbitrary fake-detection threshold.

    const elaScore = 10;


    // --------------------------------------------------
    // 4. FORGERY — 15 POINTS
    // --------------------------------------------------
    // Forgery features are supporting evidence.
    // They are not strong enough to be an independent
    // AI classifier according to our validation.

    const forgeryScore = 15;




    // --------------------------------------------------
    // FINAL SCORE
    // --------------------------------------------------

    const trustScore = Math.round(
        clamp(
            aiScore +
            metadataScore +
            elaScore +
            forgeryScore
        )
    );


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

    let recommendation;

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


    return {

        trustScore,

        riskLevel,

        recommendation,

        scoreBreakdown: {

            ai: Number(aiScore.toFixed(2)),

            metadata:
                Number(metadataScore.toFixed(2)),

            ela: elaScore,

            forgery: forgeryScore,


        }

    };
};