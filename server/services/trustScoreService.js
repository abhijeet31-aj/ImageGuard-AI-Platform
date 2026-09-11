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
        Number(aiDetection?.ai_probability ?? 0),
        0,
        1
    );

    const aiScore = (1 - aiProbability) * 60;


    // --------------------------------------------------
    // 2. METADATA — 15 POINTS
    // --------------------------------------------------

    const integrityScore = clamp(
        Number(metadataAnalysis?.integrityScore ?? 0),
        0,
        100
    );

    const metadataScore =
        (integrityScore / 100) * 15;


    // --------------------------------------------------
    // 3. ELA — 10 POINTS
    // --------------------------------------------------
    // ELA is supporting forensic evidence.
    // It is not used as an independent AI classifier.

    const elaScore = 10;


    // --------------------------------------------------
    // 4. OPENCV FORENSICS — 15 POINTS
    // --------------------------------------------------
    // OpenCV forensic features are currently treated
    // as supporting evidence rather than a standalone
    // authenticity classifier.

    const forgeryScore = 15;


    // --------------------------------------------------
    // FINAL TRUST SCORE
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

            metadata: Number(
                metadataScore.toFixed(2)
            ),

            ela: elaScore,

            forgery: forgeryScore

        }

    };
};