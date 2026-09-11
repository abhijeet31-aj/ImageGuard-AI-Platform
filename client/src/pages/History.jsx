import { useEffect, useState } from "react";
import api from "../services/api";
import "./History.css";

function toPercent(value) {
    const number = Number(value);

    if (Number.isNaN(number)) return null;

    return Math.round((number <= 1 ? number * 100 : number) * 10) / 10;
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
                    error.response?.data?.message || "Could not load analysis history."
                );
            } finally {
                setLoading(false);
            }
        };

        loadHistory();
    }, []);

    return (
        <main className="history-page">
            <header className="history-header">
                <button className="history-back-button" onClick={onBack}>
                    ← Back to Dashboard
                </button>

                <div className="history-brand">
                    <span>IG</span>
                    ImageGuard
                </div>
            </header>

            <section className="history-content">
                <p className="history-eyebrow">SAVED RESULTS</p>
                <h1>Analysis History</h1>
                <p className="history-description">
                    Review your previous combined AI and forensic analysis results.
                </p>

                {loading && <p className="history-state">Loading history...</p>}

                {message && <p className="history-state error">{message}</p>}

                {!loading && !message && analyses.length === 0 && (
                    <div className="empty-history">
                        <h2>No analysis yet</h2>
                        <p>Upload an image to create your first analysis result.</p>
                    </div>
                )}

                <div className="history-grid">
                    {analyses.map((analysis) => {
                        const fusion = analysis.report?.fusionAnalysis || {};
                        const confidence = toPercent(fusion.confidence);
                        const classification = fusion.prediction || "Not available";

                        const imageUrl = analysis.image
                            ? `http://localhost:5000/uploads/${analysis.image}`
                            : "";

                        return (
                            <article className="history-card" key={analysis._id}>
                                <div className="history-image">
                                    {imageUrl ? (
                                        <img src={imageUrl} alt="Analyzed upload" />
                                    ) : (
                                        <span>Image unavailable</span>
                                    )}
                                </div>

                                <div className="history-card-content">
                                    <p className="history-date">
                                        {new Date(analysis.createdAt).toLocaleDateString()}
                                    </p>

                                    <h2>{classification}</h2>

                                    <p>
                                        Confidence:{" "}
                                        <strong>
                                            {confidence === null ? "Not available" : `${confidence}%`}
                                        </strong>
                                    </p>

                                    <button
                                        onClick={() => onViewResult(analysis, imageUrl)}
                                        className="view-result-button"
                                    >
                                        View Result →
                                    </button>
                                </div>
                            </article>
                        );
                    })}
                </div>
            </section>
        </main>
    );
}

export default History;