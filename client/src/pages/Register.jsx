import { useState } from "react";
import api from "../services/api";
import "./Login.css";

function Register({ onGoLogin }) {
    const [name, setName] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [message, setMessage] = useState("");
    const [loading, setLoading] = useState(false);

    const handleRegister = async (event) => {
        event.preventDefault();
        setMessage("");

        if (password !== confirmPassword) {
            setMessage("Passwords do not match.");
            return;
        }

        setLoading(true);

        try {
            const response = await api.post("/auth/register", {
                name,
                email,
                password,
            });

            setMessage(response.data.message || "Account created successfully.");
        } catch (error) {
            setMessage(
                error.response?.data?.message || "Registration failed. Please try again."
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
                        Create your account to analyze image authenticity and save your
                        results securely.
                    </p>
                </div>

                <p className="intro-footer">Secure analysis. Clear evidence.</p>
            </section>

            <section className="login-panel">
                <div className="login-card">
                    <div className="mobile-brand">
                        <div className="brand-mark">IG</div>
                        <span>ImageGuard</span>
                    </div>

                    <p className="eyebrow">CREATE ACCOUNT</p>
                    <h2>Get started with ImageGuard</h2>
                    <p className="card-description">
                        Create an account to save and review your analysis history.
                    </p>

                    <form onSubmit={handleRegister}>
                        <label htmlFor="name">Full name</label>
                        <input
                            id="name"
                            type="text"
                            value={name}
                            onChange={(event) => setName(event.target.value)}
                            placeholder="Enter your name"
                            required
                        />

                        <label htmlFor="email">Email address</label>
                        <input
                            id="email"
                            type="email"
                            value={email}
                            onChange={(event) => setEmail(event.target.value)}
                            placeholder="name@example.com"
                            required
                        />

                        <label htmlFor="password">Password</label>
                        <input
                            id="password"
                            type="password"
                            value={password}
                            onChange={(event) => setPassword(event.target.value)}
                            placeholder="Create a password"
                            required
                        />

                        <label htmlFor="confirmPassword">Confirm password</label>
                        <input
                            id="confirmPassword"
                            type="password"
                            value={confirmPassword}
                            onChange={(event) => setConfirmPassword(event.target.value)}
                            placeholder="Confirm your password"
                            required
                        />

                        <button className="login-button" type="submit" disabled={loading}>
                            {loading ? "Creating account..." : "Create account"}
                            {!loading && <span>→</span>}
                        </button>
                    </form>

                    {message && (
                        <p
                            className={`login-message ${message.toLowerCase().includes("success") ? "success" : "error"
                                }`}
                        >
                            {message}
                        </p>
                    )}

                    <p className="auth-switch">
                        Already have an account?{" "}
                        <button type="button" onClick={onGoLogin}>
                            Sign in
                        </button>
                    </p>
                </div>
            </section>
        </main>
    );
}

export default Register;