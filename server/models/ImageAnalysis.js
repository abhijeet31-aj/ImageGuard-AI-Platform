import mongoose from "mongoose";

const imageAnalysisSchema = new mongoose.Schema(
    {
        uploadedBy: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "User",
            required: true,
        },

        image: {
            type: String,
            required: true,
        },
        imageHash: {
            type: String,
            required: true,
        },
        status: {
            type: String,
            enum: ["pending", "processing", "completed", "failed"],
            default: "pending",
        },

        trustScore: {
            type: Number,
            default: null,
        },

        report: {
            type: Object,
            default: {},
        },
    },
    {
        timestamps: true,
        minimize: false,
    }
);

const ImageAnalysis = mongoose.model(
    "ImageAnalysis",
    imageAnalysisSchema
);

export default ImageAnalysis;