import { useState } from "react";
import api from "../services/api";
import "./Result.css";

function toPercent(value, decimals = 0) {
    const number = Number(value);

    if (Number.isNaN(number)) return null;

    const percentage = number <= 1 ? number * 100 : number;

    return Number(percentage.toFixed(decimals));
}

function confidenceLevel(confidence) {
    if (confidence === null) return "Not available";
    if (confidence >= 80) return "High";
    if (confidence >= 50) return "Medium";

    return "Low";
}

function trustLevel(score) {
    if (score === null) return "Not available";
    if (score >= 70) return "High";
    if (score >= 40) return "Medium";

    return "Low";
}

function displayClassification(classification) {
    switch (classification) {
        case "Authentic":
            return "Authentic";

        case "AI Generated":
            return "AI Generated";

        case "Manipulated":
            return "Digitally Manipulated";

        case "Partially AI-Edited":
            return "Partially AI Edited";

        case "Needs Review":
            return "Needs Review";

        default:
            return classification || "Needs Review";
    }
}

function getVerdict(classification) {
    switch (classification) {
        case "Authentic":
            return {
                tone: "ig-authentic",
                badge: "AUTHENTIC",
                title: "This image is likely authentic",
                description:
                    "ImageGuard found no strong signs of AI-generated content or significant digital manipulation.",
            };

        case "AI Generated":
            return {
                tone: "ig-ai-warning",
                badge: "AI GENERATED",
                title: "This image is likely AI generated",
                description:
                    "ImageGuard detected strong patterns associated with AI-generated image content.",
            };

        case "Manipulated":
            return {
                tone: "ig-ai-warning",
                badge: "DIGITALLY EDITED",
                title: "This image is likely digitally edited",
                description:
                    "ImageGuard detected signals suggesting that the image may have been digitally altered.",
            };

        case "Partially AI-Edited":
            return {
                tone: "ig-ai-warning",
                badge: "AI EDITED",
                title: "This image may be partially AI edited",
                description:
                    "Some areas of the image show signals that may indicate localized AI-based editing.",
            };

        default:
            return {
                tone: "ig-needs-review",
                badge: "NEEDS REVIEW",
                title: "This image needs further review",
                description:
                    "The available signals are not strong enough to provide a reliable conclusion.",
            };
    }
}

function getRecommendation(classification, backendRecommendation) {
    if (backendRecommendation) {
        return backendRecommendation;
    }

    switch (classification) {
        case "Authentic":
            return "No significant signs of AI generation or digital manipulation were detected. The image can be considered likely authentic.";

        case "AI Generated":
            return "This image shows strong signs of AI-generated content. Verify the original source before using or sharing it.";

        case "Manipulated":
            return "This image may have been digitally altered. Verify the original image or source before relying on it.";

        case "Partially AI-Edited":
            return "Some parts of this image may have been modified using AI. Review the original source before using it as evidence.";

        default:
            return "The available signals are inconclusive. Consider verifying the image through its original source or additional evidence.";
    }
}

function getAIContent(aiLikelihood) {
    if (aiLikelihood === null) {
        return {
            value: "Not available",
            description: "AI content could not be determined.",
        };
    }

    if (aiLikelihood >= 70) {
        return {
            value: "High",
            description:
                "Strong signals associated with AI-generated content were detected.",
        };
    }

    if (aiLikelihood >= 40) {
        return {
            value: "Moderate",
            description:
                "Some signals associated with AI-generated content were detected.",
        };
    }

    return {
        value: "Low",
        description:
            "Only limited signals associated with AI-generated content were detected.",
    };
}

function getManipulationContent(manipulationLikelihood) {
    if (manipulationLikelihood === null) {
        return {
            value: "Not available",
            description: "Digital manipulation could not be determined.",
        };
    }

    if (manipulationLikelihood >= 70) {
        return {
            value: "Detected",
            description:
                "The analysis found signals suggesting digital alteration.",
        };
    }

    if (manipulationLikelihood >= 40) {
        return {
            value: "Possible",
            description:
                "Some signals may indicate digital alteration.",
        };
    }

    return {
        value: "Not detected",
        description:
            "No strong manipulation signals were detected.",
    };
}

function Result({
    analysis,
    imagePreview,
    onBack,
    onAnalyzeAgain,
}) {
    const [deepScanLoading, setDeepScanLoading] = useState(false);
    const [deepScanError, setDeepScanError] = useState(null);
    const [deepScanReport, setDeepScanReport] = useState(null);

    /*
     * Keep the original working image flow:
     * Upload.jsx creates the preview and App.jsx passes it
     * to Result.jsx as imagePreview.
     */
    const report =
        deepScanReport ||
        analysis?.report ||
        {};

    /*
     * FinalAnalysis is the current 4-class result.
     * FusionAnalysis is only used for older records.
     */
    const finalAnalysis =
        report.finalAnalysis || {};

    const fusionAnalysis =
        report.fusionAnalysis || {};

    const classification =
        finalAnalysis.prediction ||
        fusionAnalysis.prediction ||
        "Needs Review";

    const verdict = getVerdict(classification);

    /*
     * AI LIKELIHOOD
     */
    const aiLikelihood = toPercent(
        finalAnalysis.aiLikelihood ??
        finalAnalysis.ai_likelihood ??
        report.aiDetection?.aiProbability ??
        report.aiDetection?.ai_probability ??
        fusionAnalysis.fusion_probability,
        1
    );

    /*
     * CONFIDENCE
     */
    const confidence = toPercent(
        finalAnalysis.confidence ??
        finalAnalysis.confidenceScore ??
        fusionAnalysis.confidence,
        1
    );

    /*
     * MANIPULATION
     * This is used only as a simple user-facing indicator.
     */
    const manipulationLikelihood = toPercent(
        finalAnalysis.manipulationLikelihood ??
        finalAnalysis.manipulation_likelihood ??
        report.manipulationAnalysis?.manipulationProbability ??
        report.manipulationAnalysis?.manipulation_probability,
        1
    );

    /*
     * TRUST SCORE
     */
    const trustScore = toPercent(
        report.trustScore ??
        analysis?.trustScore,
        0
    );

    const currentTrustLevel = trustLevel(trustScore);

    const aiContent = getAIContent(aiLikelihood);

    const manipulationContent =
        getManipulationContent(manipulationLikelihood);

    /*
     * Recommendation from backend is preferred.
     * Otherwise frontend fallback is used.
     */
    const recommendation = getRecommendation(classification);

    /*
     * DEEP SCAN
     */
    const handleDeepScan = async () => {
        if (!analysis?._id) return;

        setDeepScanLoading(true);
        setDeepScanError(null);

        try {
            const token = localStorage.getItem("token");

            const response = await api.post(
                `/images/${analysis._id}/deep-scan`,
                {},
                {
                    headers: {
                        Authorization: `Bearer ${token}`,
                    },
                }
            );

            /*
             * Backend returns:
             * response.data.analysis
             */
            setDeepScanReport(
                response.data.analysis.report
            );

        } catch (error) {
            setDeepScanError(
                error?.response?.data?.message ||
                "Deep scan failed. Please try again."
            );
        } finally {
            setDeepScanLoading(false);
        }
    };

    return (
        <main className="ig-result-page">

            {/* HEADER */}
            <header className="ig-result-header">

                <button
                    className="ig-back-button"
                    onClick={onBack}
                >
                    ← Back to Dashboard
                </button>

                <div className="ig-result-brand">
                    <span className="ig-brand-mark">
                        IG
                    </span>

                    ImageGuard
                </div>

            </header>

            {/* CONTENT */}
            <section className="ig-result-content">

                <p className="ig-result-tagline">
                    AI-BASED IMAGE AUTHENTICITY ANALYSIS AND TRUST SCORE GENERATION
                </p>

                <h1>Image Analysis Result</h1>

                <div className="ig-analysis-shell">

                    {/* IMAGE PREVIEW */}
                    <section className="ig-image-panel">

                        <div className="ig-image-panel-heading">
                            <span className="ig-pulse-dot"></span>
                            Evidence Preview
                        </div>

                        {imagePreview ? (
                            <img
                                src={imagePreview}
                                alt="Analyzed image"
                            />
                        ) : (
                            <div className="ig-preview-empty">
                                Image preview unavailable
                            </div>
                        )}

                    </section>

                    {/* RESULT */}
                    <section className="ig-verdict-panel">

                        {/* MAIN VERDICT */}
                        <div
                            className={`ig-verdict ${verdict.tone}`}
                        >

                            <div>
                                <span>
                                    IMAGEGUARD VERDICT
                                </span>

                                <h2>
                                    {verdict.title}
                                </h2>

                                <p className="ig-verdict-description">
                                    {verdict.description}
                                </p>
                            </div>

                            <div className="ig-percentage">

                                <strong>
                                    {confidence === null
                                        ? "—"
                                        : `${confidence}%`}
                                </strong>

                                <small>
                                    {verdict.badge}
                                </small>

                            </div>

                        </div>

                        {/* SIMPLE RESULT DETAILS */}
                        <div className="ig-result-details">

                            {/* AI LIKELIHOOD */}
                            <div className="ig-detail-row">

                                <span>
                                    AI Likelihood
                                </span>

                                <strong>
                                    {aiLikelihood === null
                                        ? "Not available"
                                        : `${aiLikelihood}%`}
                                </strong>

                            </div>

                            {/* CONFIDENCE */}
                            <div className="ig-detail-row">

                                <span>
                                    Confidence
                                </span>

                                <strong>
                                    {confidence === null
                                        ? "Not available"
                                        : `${confidence}% (${confidenceLevel(
                                            confidence
                                        )})`}
                                </strong>

                            </div>

                            {/* CLASSIFICATION */}
                            <div className="ig-detail-row">

                                <span>
                                    Classification
                                </span>

                                <strong>
                                    {displayClassification(
                                        classification
                                    )}
                                </strong>

                            </div>

                            {/* TRUST SCORE */}
                            {trustScore !== null && (
                                <div className="ig-detail-row">

                                    <span>
                                        Trust Score
                                    </span>

                                    <strong>
                                        {trustScore}/100
                                    </strong>

                                </div>
                            )}

                            {/* TRUST LEVEL */}
                            {trustScore !== null && (
                                <div className="ig-detail-row">

                                    <span>
                                        Trust Level
                                    </span>

                                    <strong>
                                        {currentTrustLevel}
                                    </strong>

                                </div>
                            )}

                        </div>

                        {/* WHY THIS RESULT */}
                        <div className="ig-explanation-panel">

                            <h3>
                                Why this result?
                            </h3>

                            <p>
                                {verdict.description}
                            </p>

                            <div className="ig-simple-indicators">

                                {/* AI CONTENT */}
                                <div className="ig-simple-card">

                                    <span>
                                        AI Content
                                    </span>

                                    <strong>
                                        {aiContent.value}
                                    </strong>

                                    <p>
                                        {aiContent.description}
                                    </p>

                                </div>

                                {/* MANIPULATION */}
                                <div className="ig-simple-card">

                                    <span>
                                        Image Manipulation
                                    </span>

                                    <strong>
                                        {manipulationContent.value}
                                    </strong>

                                    <p>
                                        {manipulationContent.description}
                                    </p>

                                </div>

                            </div>

                        </div>

                        {/* RECOMMENDATION */}
                        <div className="ig-result-details">

                            <div className="ig-detail-row ig-recommendation">

                                <span>
                                    Recommendation
                                </span>

                                <strong>
                                    {recommendation}
                                </strong>

                            </div>

                        </div>

                        {/* DEEP SCAN */}
                        <div className="ig-deep-scan-panel">

                            <button
                                className="ig-deep-scan-button"
                                onClick={handleDeepScan}
                                disabled={deepScanLoading}
                            >
                                {deepScanLoading
                                    ? "Scanning regions…"
                                    : "Run Deep Scan (check for partial AI edits)"}
                            </button>

                            {deepScanError && (
                                <p className="ig-deep-scan-error">
                                    {deepScanError}
                                </p>
                            )}

                            {report.tileAnalysis && (
                                <div className="ig-deep-scan-results">

                                    <div className="ig-detail-row">

                                        <span>
                                            Deep Scan Result
                                        </span>

                                        <strong>
                                            {
                                                report.tileAnalysis
                                                    .tileVerdict
                                            }
                                        </strong>

                                    </div>

                                    <div className="ig-detail-row">

                                        <span>
                                            Regions Checked
                                        </span>

                                        <strong>
                                            {
                                                report.tileAnalysis
                                                    .tileCount
                                            }
                                        </strong>

                                    </div>

                                    {report.suspiciousRegions
                                        ?.length > 0 && (
                                            <div className="ig-detail-row">

                                                <span>
                                                    Suspicious Regions Found
                                                </span>

                                                <strong>
                                                    {
                                                        report
                                                            .suspiciousRegions
                                                            .length
                                                    }
                                                </strong>

                                            </div>
                                        )}

                                    {report.finalAnalysis
                                        ?.prediction ===
                                        "Partially AI-Edited" && (
                                            <p className="ig-deep-scan-warning">
                                                Some regions of this image
                                                may contain localized AI
                                                edits. Treat the image
                                                carefully and verify its
                                                original source.
                                            </p>
                                        )}

                                </div>
                            )}

                        </div>

                        {/* ANALYZE AGAIN */}
                        <button
                            className="ig-analyze-button"
                            onClick={onAnalyzeAgain}
                        >
                            Analyze New Image
                            <span>→</span>
                        </button>

                    </section>

                </div>

            </section>

        </main>
    );
}

export default Result;