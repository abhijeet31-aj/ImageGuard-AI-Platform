import { useEffect, useState } from "react";
import api from "../services/api";
import "./History.css";

function toPercent(value) {
    const number = Number(value);

    if (Number.isNaN(number)) return null;

    return Math.round(
        (number <= 1 ? number * 100 : number) * 10
    ) / 10;
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

function History({ onBack, onViewResult }) {
    const [analyses, setAnalyses] = useState([]);
    const [loading, setLoading] = useState(true);
    const [message, setMessage] = useState("");

    useEffect(() => {
        const loadHistory = async () => {
            try {
                const token = localStorage.getItem("token");

                const response = await api.get("/images/history", {
                    headers: {
                        Authorization: `Bearer ${token}`,
                    },
                });

                setAnalyses(response.data.analyses || []);
            } catch (error) {
                setMessage(
                    error.response?.data?.message ||
                    "Could not load analysis history."
                );
            } finally {
                setLoading(false);
            }
        };

        loadHistory();
    }, []);

    return (
        <main className="history-page">

            {/* HEADER */}
            <header className="history-header">

                <button
                    className="history-back-button"
                    onClick={onBack}
                >
                    ← Back to Dashboard
                </button>

                <div className="history-brand">
                    <span>IG</span>
                    ImageGuard
                </div>

            </header>

            {/* CONTENT */}
            <section className="history-content">

                <p className="history-eyebrow">
                    SAVED RESULTS
                </p>

                <h1>Analysis History</h1>

                <p className="history-description">
                    Review your previous image authenticity assessments.
                </p>

                {/* LOADING */}
                {loading && (
                    <p className="history-state">
                        Loading history...
                    </p>
                )}

                {/* ERROR */}
                {message && (
                    <p className="history-state error">
                        {message}
                    </p>
                )}

                {/* EMPTY */}
                {!loading &&
                    !message &&
                    analyses.length === 0 && (
                        <div className="empty-history">
                            <h2>No analysis yet</h2>

                            <p>
                                Upload an image to create your
                                first analysis result.
                            </p>
                        </div>
                    )}

                {/* HISTORY GRID */}
                {!loading &&
                    !message &&
                    analyses.length > 0 && (
                        <div className="history-grid">

                            {analyses.map((analysis) => {

                                const report =
                                    analysis.report || {};

                                /*
                                 * Current 4-class result first.
                                 * Old fusion result is fallback.
                                 */
                                const finalAnalysis =
                                    report.finalAnalysis || {};

                                const fusionAnalysis =
                                    report.fusionAnalysis || {};

                                const rawClassification =
                                    finalAnalysis.prediction ||
                                    fusionAnalysis.prediction ||
                                    "Needs Review";

                                const classification =
                                    displayClassification(
                                        rawClassification
                                    );

                                /* Confidence */
                                const confidence = toPercent(
                                    finalAnalysis.confidence ??
                                    fusionAnalysis.confidence
                                );

                                /* AI Likelihood */
                                const aiLikelihood =
                                    toPercent(
                                        finalAnalysis.aiLikelihood ??
                                        finalAnalysis.ai_likelihood ??
                                        report.aiDetection
                                            ?.aiProbability ??
                                        report.aiDetection
                                            ?.ai_probability
                                    );

                                /* Trust Score */
                                const trustScore = toPercent(
                                    report.trustScore ??
                                    analysis.trustScore
                                );

                                /* Uploaded image */
                                const imageUrl = analysis.image
                                    ? `http://localhost:5000/uploads/${analysis.image}`
                                    : "";

                                return (
                                    <article
                                        className="history-card"
                                        key={analysis._id}
                                    >

                                        {/* IMAGE */}
                                        <div className="history-image">

                                            {imageUrl ? (
                                                <img
                                                    src={imageUrl}
                                                    alt="Analyzed upload"
                                                />
                                            ) : (
                                                <span>
                                                    Image unavailable
                                                </span>
                                            )}

                                        </div>

                                        {/* CARD CONTENT */}
                                        <div className="history-card-content">

                                            <p className="history-date">
                                                {new Date(
                                                    analysis.createdAt
                                                ).toLocaleDateString()}
                                            </p>

                                            <h2>
                                                {classification}
                                            </h2>

                                            {/* CONFIDENCE */}
                                            <p>
                                                Confidence:{" "}
                                                <strong>
                                                    {confidence === null
                                                        ? "Not available"
                                                        : `${confidence}%`}
                                                </strong>
                                            </p>

                                            {/* AI LIKELIHOOD */}
                                            {aiLikelihood !== null && (
                                                <p>
                                                    AI Likelihood:{" "}
                                                    <strong>
                                                        {aiLikelihood}%
                                                    </strong>
                                                </p>
                                            )}

                                            {/* TRUST SCORE */}
                                            {trustScore !== null && (
                                                <p>
                                                    Trust Score:{" "}
                                                    <strong>
                                                        {trustScore}/100
                                                    </strong>
                                                </p>
                                            )}

                                            <button
                                                onClick={() =>
                                                    onViewResult(
                                                        analysis,
                                                        imageUrl
                                                    )
                                                }
                                                className="view-result-button"
                                            >
                                                View Result →
                                            </button>

                                        </div>

                                    </article>
                                );
                            })}

                        </div>
                    )}

            </section>
        </main>
    );
}

export default History;