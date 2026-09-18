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

function Result({ analysis, imagePreview, onBack, onAnalyzeAgain }) {
    const [deepScanLoading, setDeepScanLoading] = useState(false);
    const [deepScanError, setDeepScanError] = useState(null);
    const [deepScanReport, setDeepScanReport] = useState(null);

    const report = deepScanReport || analysis?.report || {};

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

            setDeepScanReport(response.data.analysis.report);

        } catch (error) {
            setDeepScanError(
                error?.response?.data?.message ||
                "Deep scan failed. Please try again."
            );
        } finally {
            setDeepScanLoading(false);
        }
    };

    // finalAnalysis (Phase 2-4) is the real multi-class verdict —
    // Authentic / AI Generated / Manipulated / Partially AI-Edited /
    // Needs Review. fusionAnalysis is kept only as a fallback for old
    // records saved before finalAnalysis existed.
    const finalAnalysis = report.finalAnalysis || {};
    const fusionAnalysis = report.fusionAnalysis || {};

    const classification =
        finalAnalysis.prediction ||
        fusionAnalysis.prediction ||
        "Not available";

    const aiLikelihood = toPercent(
        finalAnalysis.aiLikelihood ?? fusionAnalysis.fusion_probability,
        1
    );

    const confidence = toPercent(
        finalAnalysis.confidence ?? fusionAnalysis.confidence,
        1
    );

    const manipulationLikelihood = toPercent(
        finalAnalysis.manipulationLikelihood,
        1
    );

    const VERDICT_STYLES = {
        "Authentic": { tone: "ig-authentic", badge: "AUTHENTIC", text: "This image appears likely authentic" },
        "AI Generated": { tone: "ig-ai-warning", badge: "AI GENERATED", text: "This image contains likely AI-generated content" },
        "Manipulated": { tone: "ig-ai-warning", badge: "MANIPULATED", text: "Possible image manipulation detected" },
        "Partially AI-Edited": { tone: "ig-ai-warning", badge: "AI EDITED", text: "This image may contain a localized AI edit" },
        "Needs Review": { tone: "ig-needs-review", badge: "NEEDS REVIEW", text: "Signals were inconclusive for this image" },
    };

    const verdictStyle =
        VERDICT_STYLES[classification] ||
        { tone: "ig-needs-review", badge: "NEEDS REVIEW", text: "Signals were inconclusive for this image" };

    const recommendation =
        report.recommendation ||
        finalAnalysis.recommendation ||
        "Needs Review";

    return (
        <main className="ig-result-page">
            <header className="ig-result-header">
                <button className="ig-back-button" onClick={onBack}>
                    ← Back to Dashboard
                </button>

                <div className="ig-result-brand">
                    <span className="ig-brand-mark">IG</span>
                    ImageGuard
                </div>
            </header>

            <section className="ig-result-content">
                <p className="ig-result-tagline">
                    AI-BASED IMAGE AUTHENTICITY ANALYSIS AND TRUST SCORE GENERATION
                </p>

                <h1>Image Analysis Result</h1>

                <div className="ig-analysis-shell">
                    <section className="ig-image-panel">
                        <div className="ig-image-panel-heading">
                            <span className="ig-pulse-dot"></span>
                            Evidence Preview
                        </div>

                        {imagePreview ? (
                            <img src={imagePreview} alt="Analyzed image" />
                        ) : (
                            <div className="ig-preview-empty">Image preview unavailable</div>
                        )}
                    </section>

                    <section className="ig-verdict-panel">
                        <div className={`ig-verdict ${verdictStyle.tone}`}>
                            <div>
                                <span>IMAGEGUARD VERDICT</span>
                                <h2>{verdictStyle.text}</h2>
                            </div>

                            <div className="ig-percentage">
                                <strong>
                                    {confidence === null ? "—" : `${confidence}%`}
                                </strong>
                                <small>{verdictStyle.badge}</small>
                            </div>
                        </div>

                        <div className="ig-result-details">
                            <div className="ig-detail-row">
                                <span>AI Likelihood</span>
                                <strong>
                                    {aiLikelihood === null
                                        ? "Not available"
                                        : `${aiLikelihood}%`}
                                </strong>
                            </div>

                            <div className="ig-detail-row">
                                <span>Manipulation Likelihood</span>
                                <strong>
                                    {manipulationLikelihood === null
                                        ? "Not available"
                                        : `${manipulationLikelihood}%`}
                                </strong>
                            </div>

                            <div className="ig-detail-row">
                                <span>Confidence</span>
                                <strong>
                                    {confidence === null
                                        ? "Not available"
                                        : `${confidence}% (${confidenceLevel(confidence)})`}
                                </strong>
                            </div>

                            <div className="ig-detail-row">
                                <span>Classification</span>
                                <strong>{classification}</strong>
                            </div>

                            <div className="ig-detail-row ig-recommendation">
                                <span>Recommendation</span>
                                <strong>{recommendation}</strong>
                            </div>
                        </div>

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
                                <p className="ig-deep-scan-error">{deepScanError}</p>
                            )}

                            {report.tileAnalysis && (
                                <div className="ig-deep-scan-results">
                                    <div className="ig-detail-row">
                                        <span>Deep Scan Result</span>
                                        <strong>{report.tileAnalysis.tileVerdict}</strong>
                                    </div>

                                    <div className="ig-detail-row">
                                        <span>Regions Checked</span>
                                        <strong>{report.tileAnalysis.tileCount}</strong>
                                    </div>

                                    {report.suspiciousRegions?.length > 0 && (
                                        <div className="ig-detail-row">
                                            <span>Suspicious Regions Found</span>
                                            <strong>{report.suspiciousRegions.length}</strong>
                                        </div>
                                    )}

                                    {report.finalAnalysis?.prediction ===
                                        "Partially AI-Edited" && (
                                            <p className="ig-deep-scan-warning">
                                                Some regions of this image score very
                                                differently from the rest — consistent
                                                with a localized AI edit. Treat this
                                                image as only partially verified.
                                            </p>
                                        )}
                                </div>
                            )}
                        </div>

                        <button className="ig-analyze-button" onClick={onAnalyzeAgain}>
                            Analyze New Image <span>→</span>
                        </button>
                    </section>
                </div>
            </section>
        </main>
    );
}

export default Result;