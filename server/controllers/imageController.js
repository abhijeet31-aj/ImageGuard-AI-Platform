import path from "path";
import ImageAnalysis from "../models/ImageAnalysis.js";
import { generateImageHash } from "../services/hashService.js";
import { checkDuplicateImage } from "../services/duplicateService.js";
import { extractMetadata } from "../services/metadataService.js";
import { analyzeResolution, analyzeBrightness, } from "../services/qualityService.js";

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

        const resolution = analyzeResolution(metadata);

        const brightness = await analyzeBrightness(imagePath);

        const imageHash = await generateImageHash(imagePath);

        const previousAnalysis = await checkDuplicateImage(
            req.user._id,
            imageHash
        );

        const imageAnalysis = await ImageAnalysis.create({
            uploadedBy: req.user._id,
            image: req.file.filename,
            imageHash: imageHash,
            status: "pending",
            report: {
                metadata: metadata,
                duplicateCheck: {
                    isDuplicate: previousAnalysis ? true : false,

                    previousAnalysisId: previousAnalysis
                        ? previousAnalysis._id
                        : null,

                    previousAnalysisDate: previousAnalysis
                        ? previousAnalysis.createdAt
                        : null,
                },
                aiDetection: {},
                qualityAssessment: {
                    resolution,
                    brightness,
                    contrast: {},
                    blur: {},
                },
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