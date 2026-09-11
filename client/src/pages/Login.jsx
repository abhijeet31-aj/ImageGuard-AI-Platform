import { useState } from "react";
import api from "../services/api";
import "./Login.css";

function Login({ onLogin, onRegister }) {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [message, setMessage] = useState("");
    const [loading, setLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);

    const handleLogin = async (e) => {
        e.preventDefault();

        setMessage("");
        setLoading(true);

        try {
            const response = await api.post("/auth/login", {
                email,
                password,
            });

            console.log("Login response:", response.data);

            const token = response.data.token;

            if (token) {
                localStorage.setItem("token", token);
                setMessage("Login successful!");
                onLogin();
            } else {
                setMessage("Login successful, but token not received.");
            }
        } catch (error) {
            console.error("Login error:", error);

            setMessage(
                error.response?.data?.message || "Login failed. Please try again."
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="login-page">
            <section className="login-intro">
                <div className="brand">
                    <div className="brand-mark">IG</div>
                    <span>ImageGuard</span>
                </div>

                <div className="intro-content">
                    <p className="eyebrow">AI IMAGE FORENSICS PLATFORM</p>
                    <h1>Verify every image with confidence.</h1>
                    <p className="intro-description">
                        Analyze authenticity, detect manipulation, and receive a clear
                        trust report for every image.
                    </p>

                    <div className="feature-list">
                        <div className="feature-item">
                            <span>✓</span>
                            AI-generated image detection
                        </div>
                        <div className="feature-item">
                            <span>✓</span>
                            Manipulation and metadata analysis
                        </div>
                        <div className="feature-item">
                            <span>✓</span>
                            Clear trust scores and reports
                        </div>
                    </div>
                </div>

                <p className="intro-footer">Secure analysis. Clear evidence.</p>
            </section>

            <section className="login-panel">
                <div className="login-card">
                    <div className="mobile-brand">
                        <div className="brand-mark">IG</div>
                        <span>ImageGuard</span>
                    </div>

                    <p className="eyebrow">WELCOME BACK</p>
                    <h2>Sign in to your account</h2>
                    <p className="card-description">
                        Access your image-analysis workspace and reports.
                    </p>

                    <form onSubmit={handleLogin}>
                        <label htmlFor="email">Email address</label>
                        <input
                            id="email"
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="name@example.com"
                            autoComplete="email"
                            required
                        />

                        <label htmlFor="password">Password</label>
                        <div className="password-field">
                            <input
                                id="password"
                                type={showPassword ? "text" : "password"}
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="Enter your password"
                                autoComplete="current-password"
                                required
                            />
                            <button
                                type="button"
                                className="password-toggle"
                                onClick={() => setShowPassword(!showPassword)}
                                aria-label={showPassword ? "Hide password" : "Show password"}
                            >
                                {showPassword ? "Hide" : "Show"}
                            </button>
                        </div>

                        <button className="login-button" type="submit" disabled={loading}>
                            {loading ? "Signing in..." : "Sign in"}
                            {!loading && <span>→</span>}
                        </button>
                    </form>

                    {message && (
                        <p
                            className={`login-message ${message.includes("successful") ? "success" : "error"
                                }`}
                            aria-live="polite"
                        >
                            {message}
                        </p>
                    )}

                    <p className="auth-switch">
                        New to ImageGuard?{" "}
                        <button type="button" onClick={onRegister}>
                            Create an account
                        </button>
                    </p>

                    <p className="security-note">
                        Your account and analysis history are protected.
                    </p>
                </div>
            </section>
        </main>
    );
}

export default Login;