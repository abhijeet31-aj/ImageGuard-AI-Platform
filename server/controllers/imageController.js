import path from "path";
import ImageAnalysis from "../models/ImageAnalysis.js";
import { extractMetadata } from "../services/metadataService.js";

export const uploadImage = async (req, res) => {
    try {

        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: "No image uploaded"
            });
        }

        const imagePath = path.join("uploads", req.file.filename);

        const metadata = await extractMetadata(imagePath);

        const imageAnalysis = await ImageAnalysis.create({
            uploadedBy: req.user._id,
            image: req.file.filename,
            status: "pending",
            report: {
                metadata: metadata,
                duplicateCheck: {},
                aiDetection: {},
                qualityAssessment: {},
                recommendation: ""
            }
        });

        res.status(201).json({
            success: true,
            message: "Image uploaded successfully",
            imageAnalysis
        });

    } catch (error) {
        console.log(error);
        res.status(500).json({
            success: false,
            message: "Server Error"
        });
    }
};