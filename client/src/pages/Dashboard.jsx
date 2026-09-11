import "./Dashboard.css";

function Dashboard({ onLogout, onNavigate }) {
    return (
        <main className="dashboard-page">
            <header className="dashboard-header">
                <div className="dashboard-brand">
                    <div className="dashboard-brand-mark">IG</div>
                    <span>ImageGuard</span>
                </div>

                <button className="logout-button" onClick={onLogout}>
                    Log out
                </button>
            </header>

            <section className="dashboard-content">
                <p className="dashboard-eyebrow">IMAGE FORENSICS WORKSPACE</p>
                <h1>Welcome back.</h1>
                <p className="dashboard-subtitle">
                    Start a new image verification or review your previous analysis
                    reports.
                </p>

                <div className="dashboard-actions">
                    <article className="action-card primary-card">
                        <div className="card-icon">↥</div>
                        <h2>Analyze an image</h2>
                        <p>
                            Upload an image to check its authenticity using combined AI and forensic analysis.
                        </p>
                        <button
                            type="button"
                            className="primary-action"
                            onClick={() => onNavigate("upload")}
                        >
                            Upload image <span>→</span>
                        </button>
                    </article>

                    <article className="action-card">
                        <div className="card-icon neutral-icon">▣</div>
                        <h2>Analysis history</h2>
                        <p>
                            Review previous analysis results, confidence levels, classifications, and recommendations.
                        </p>
                        <button
                            type="button"
                            className="secondary-action"
                            onClick={() => onNavigate("history")}
                        >
                            View history <span>→</span>
                        </button>
                    </article>
                </div>

                <section className="dashboard-info">
                    <div>
                        <strong>What ImageGuard checks</strong>
                        <p>AI content, manipulation signals, metadata, image quality, and duplicate uploads.</p>
                    </div>
                    <div className="status-badge">
                        <span></span>
                        System ready
                    </div>
                </section>
            </section>
        </main>
    );
}

export default Dashboard;