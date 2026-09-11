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
    const report = analysis?.report || {};

    // Only combined AI + forensic fusion result is used.
    const fusionAnalysis = report.fusionAnalysis || {};

    const aiLikelihood = toPercent(fusionAnalysis.fusion_probability, 1);
    const confidence = toPercent(fusionAnalysis.confidence, 1);
    const classification = fusionAnalysis.prediction || "Not available";

    const isAiGenerated = /ai|fake|generated|synthetic/i.test(classification);

    const verdictText = isAiGenerated
        ? "This image is likely created by AI"
        : "This image appears likely authentic";

    const recommendation =
        report.recommendation ||
        "Review this image before using it as verified evidence.";

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
                        <div
                            className={`ig-verdict ${isAiGenerated ? "ig-ai-warning" : "ig-authentic"
                                }`}
                        >
                            <div>
                                <span>IMAGEGUARD VERDICT</span>
                                <h2>{verdictText}</h2>
                            </div>

                            <div className="ig-percentage">
                                <strong>
                                    {confidence === null ? "—" : `${confidence}%`}
                                </strong>
                                <small>{isAiGenerated ? "AI" : "AUTHENTIC"}</small>
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