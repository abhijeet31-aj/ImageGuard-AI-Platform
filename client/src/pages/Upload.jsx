import { useState } from "react";
import api from "../services/api";
import "./Upload.css";

function Upload({ onBack, onUploadComplete }) {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState("");
    const [message, setMessage] = useState("");
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event) => {
        const selectedFile = event.target.files?.[0];

        if (!selectedFile) return;

        if (!selectedFile.type.startsWith("image/")) {
            setMessage("Please select a valid image file.");
            return;
        }

        setMessage("");
        setFile(selectedFile);

        const reader = new FileReader();

        reader.onload = () => {
            setPreview(reader.result);
        };

        reader.onerror = () => {
            setMessage("Image preview could not be loaded.");
        };

        reader.readAsDataURL(selectedFile);
    };

    const removeSelectedFile = () => {
        setFile(null);
        setPreview("");
        setMessage("");
    };

    const handleUpload = async (event) => {
        event.preventDefault();

        if (!file) {
            setMessage("Please select an image before starting analysis.");
            return;
        }

        setLoading(true);
        setMessage("");

        const formData = new FormData();
        formData.append("image", file);
        formData.append("category", "general");

        try {
            const token = localStorage.getItem("token");

            const response = await api.post("/images/upload", formData, {
                headers: {
                    Authorization: `Bearer ${token}`,
                },
            });

            console.log("ANALYSIS DATA:", response.data.imageAnalysis);

            onUploadComplete(response.data.imageAnalysis, preview);
        } catch (error) {
            console.error("Image upload error:", error);

            setMessage(
                error.response?.data?.message ||
                "Image analysis failed. Please try again."
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="upload-page">
            <header className="upload-header">
                <button className="back-button" onClick={onBack}>
                    ← Back to dashboard
                </button>

                <div className="upload-brand">
                    <div className="upload-brand-mark">IG</div>
                    <span>ImageGuard</span>
                </div>
            </header>

            <section className="upload-content">
                <p className="upload-eyebrow">NEW IMAGE ANALYSIS</p>
                <h1>Upload an image to verify.</h1>
                <p className="upload-description">
                    Upload an image for authenticity and AI-generation analysis.
                </p>

                <form className="upload-form" onSubmit={handleUpload}>
                    <label className="file-drop-zone" htmlFor="image-upload">
                        {preview ? (
                            <img
                                className="image-preview"
                                src={preview}
                                alt="Selected preview"
                            />
                        ) : (
                            <div className="upload-placeholder">
                                <div className="upload-icon">↥</div>
                                <strong>Choose an image</strong>
                                <span>JPG, JPEG, PNG, or WEBP supported</span>
                            </div>
                        )}

                        <input
                            id="image-upload"
                            type="file"
                            accept="image/*"
                            onChange={handleFileChange}
                            hidden
                        />
                    </label>

                    {file && (
                        <div className="selected-file">
                            <span>Selected image</span>
                            <strong>{file.name}</strong>
                            <button type="button" onClick={removeSelectedFile}>
                                Remove
                            </button>
                        </div>
                    )}



                    <button className="analyze-button" type="submit" disabled={loading}>
                        {loading ? "Analyzing image..." : "Start analysis"}
                        {!loading && <span>→</span>}
                    </button>

                    {message && (
                        <p className="upload-message" aria-live="polite">
                            {message}
                        </p>
                    )}
                </form>
            </section>

            {loading && (
                <div className="analysis-overlay" role="status" aria-live="polite">
                    <div className="analysis-loader-card">
                        <div className="analysis-loader-ring">
                            <div className="analysis-loader-core">IG</div>
                        </div>

                        <p className="analysis-loader-label">IMAGEGUARD ANALYSIS</p>
                        <h2>Analyzing your image...</h2>
                        <p>
                            Our combined AI and forensic model is checking image authenticity.
                        </p>

                        <div className="analysis-loader-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                </div>
            )}
        </main>
    );
}

export default Upload;