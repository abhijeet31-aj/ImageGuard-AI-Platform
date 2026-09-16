import path from "path";

import ImageAnalysis from "../models/ImageAnalysis.js";

import { generateImageHash } from "../services/hashService.js";
import { checkDuplicateImage } from "../services/duplicateService.js";
import { extractMetadata } from "../services/metadataService.js";

import {
    analyzeResolution,
    analyzeBrightness,
} from "../services/qualityService.js";

import { sendImageToPython, sendImageForDeepScan } from "../services/pythonService.js";

import {
    calculateTrustScore
} from "../services/trustScoreService.js";

export const uploadImage = async (req, res) => {

    try {

        // --------------------------------------------------
        // 1. CHECK IMAGE
        // --------------------------------------------------

        if (!req.file) {

            return res.status(400).json({
                success: false,
                message: "No image uploaded"
            });

        }


        // --------------------------------------------------
        // 2. IMAGE PATH
        // --------------------------------------------------

        const imagePath = path.join(
            "uploads",
            req.file.filename
        );


        // --------------------------------------------------
        // 3. CATEGORY
        // --------------------------------------------------

        const category =
            req.body.category || "unknown";


        // --------------------------------------------------
        // 4. METADATA
        // --------------------------------------------------

        const metadata =
            await extractMetadata(imagePath);


        // --------------------------------------------------
        // 5. IMAGE QUALITY
        // --------------------------------------------------

        const resolution =
            analyzeResolution(metadata);

        const brightness =
            await analyzeBrightness(imagePath);


        // --------------------------------------------------
        // 6. IMAGE HASH
        // --------------------------------------------------

        const imageHash =
            await generateImageHash(imagePath);


        // --------------------------------------------------
        // 7. DUPLICATE CHECK
        // --------------------------------------------------

        const previousAnalysis =
            await checkDuplicateImage(
                req.user._id,
                imageHash
            );


        // --------------------------------------------------
        // 8. PYTHON IMAGE ANALYSIS
        // --------------------------------------------------

        const pythonAnalysis =
            await sendImageToPython(
                imagePath,
                category
            );



        const duplicateCheck = {
            isDuplicate: previousAnalysis
                ? true
                : false,

            previousAnalysisId: previousAnalysis
                ? previousAnalysis._id
                : null,

            previousAnalysisDate: previousAnalysis
                ? previousAnalysis.createdAt
                : null
        };



        const trustResult = calculateTrustScore({

            fusionAnalysis:
                pythonAnalysis.fusionAnalysis,

            manipulationAnalysis:
                pythonAnalysis.manipulationAnalysis,

            metadataAnalysis:
                pythonAnalysis.metadataAnalysis,

            finalAnalysis:
                pythonAnalysis.finalAnalysis,

        });



        // --------------------------------------------------
        // 9. CREATE ANALYSIS RECORD
        // --------------------------------------------------

        const imageAnalysis =
            await ImageAnalysis.create({

                uploadedBy: req.user._id,

                image: req.file.filename,

                imageHash: imageHash,

                status: "completed",

                trustScore: trustResult.trustScore,

                report: {

                    metadata: metadata,

                    duplicateCheck: duplicateCheck,

                    aiDetection:
                        pythonAnalysis.aiDetection,

                    metadataAnalysis:
                        pythonAnalysis.metadataAnalysis,

                    elaMetrics:
                        pythonAnalysis.elaMetrics,

                    forgeryAnalysis:
                        pythonAnalysis.forgeryAnalysis,

                    manipulationAnalysis:
                        pythonAnalysis.manipulationAnalysis,

                    fusionAnalysis:
                        pythonAnalysis.fusionAnalysis,

                    finalAnalysis:
                        pythonAnalysis.finalAnalysis,

                    qualityAssessment: {

                        resolution,

                        brightness,

                        contrast: {},

                        blur: {},

                    },

                    trustScore:
                        trustResult.trustScore,

                    riskLevel:
                        trustResult.riskLevel,

                    scoreBreakdown:
                        trustResult.scoreBreakdown,

                    recommendation:
                        trustResult.recommendation

                }

            });


        // --------------------------------------------------
        // 10. RESPONSE
        // --------------------------------------------------

        return res.status(201).json({

            success: true,

            message:
                "Image analyzed successfully",

            imageAnalysis

        });


    } catch (error) {

        console.log(
            "Image analysis error:",
            error
        );

        return res.status(500).json({

            success: false,

            message:
                error.message ||
                "Server Error"

        });

    }

};


// --------------------------------------------------
// GET USER ANALYSIS HISTORY
// --------------------------------------------------

export const getAnalysisHistory = async (req, res) => {
    try {
        const HISTORY_DAYS = 30;
        const MAX_HISTORY_ITEMS = 20;

        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - HISTORY_DAYS);

        const analyses = await ImageAnalysis.find({
            uploadedBy: req.user._id,
            createdAt: { $gte: cutoffDate },
        })
            .sort({ createdAt: -1 })
            .limit(MAX_HISTORY_ITEMS);

        return res.status(200).json({
            success: true,
            count: analyses.length,
            analyses,
        });

    } catch (error) {

        console.log(
            "History fetch error:",
            error
        );

        return res.status(500).json({
            success: false,
            message:
                error.message ||
                "Failed to fetch analysis history",
        });
    }
};


// --------------------------------------------------
// GET SINGLE ANALYSIS REPORT
// --------------------------------------------------

export const getAnalysisById = async (req, res) => {
    try {
        const analysis = await ImageAnalysis.findOne({
            _id: req.params.id,
            uploadedBy: req.user._id,
        });

        if (!analysis) {
            return res.status(404).json({
                success: false,
                message: "Analysis not found",
            });
        }

        return res.status(200).json({
            success: true,
            analysis,
        });

    } catch (error) {

        console.log(
            "Analysis fetch error:",
            error
        );

        return res.status(500).json({
            success: false,
            message:
                error.message ||
                "Failed to fetch analysis",
        });
    }
};


// --------------------------------------------------
// DEEP SCAN — Phase 3 (on-demand patch/tile pipeline)
// --------------------------------------------------
// Triggered only when the user explicitly clicks "Deep Scan" on an
// existing result — never runs automatically during upload, since
// tiling is noticeably slower than the default analysis.

export const deepScanImage = async (req, res) => {
    try {
        const analysis = await ImageAnalysis.findOne({
            _id: req.params.id,
            uploadedBy: req.user._id,
        });

        if (!analysis) {
            return res.status(404).json({
                success: false,
                message: "Analysis not found",
            });
        }

        const imagePath = path.join(
            "uploads",
            analysis.image
        );

        const deepScanResult = await sendImageForDeepScan(
            imagePath,
            analysis.report?.finalAnalysis || null
        );

        // --------------------------------------------------
        // Update the stored report with the refined findings
        // --------------------------------------------------

        if (deepScanResult.finalAnalysis) {
            analysis.report.finalAnalysis = deepScanResult.finalAnalysis;
        }

        analysis.report.suspiciousRegions =
            deepScanResult.suspiciousRegions || [];

        analysis.report.tileAnalysis =
            deepScanResult.tileAnalysis || null;


        // --------------------------------------------------
        // Recompute trust score using the refined finalAnalysis,
        // keeping the same underlying fusion/manipulation signals.
        // --------------------------------------------------

        const trustResult = calculateTrustScore({
            fusionAnalysis: analysis.report.fusionAnalysis,
            manipulationAnalysis: analysis.report.manipulationAnalysis,
            metadataAnalysis: analysis.report.metadataAnalysis,
            finalAnalysis: analysis.report.finalAnalysis,
        });

        analysis.trustScore = trustResult.trustScore;
        analysis.report.trustScore = trustResult.trustScore;
        analysis.report.riskLevel = trustResult.riskLevel;
        analysis.report.scoreBreakdown = trustResult.scoreBreakdown;
        analysis.report.recommendation = trustResult.recommendation;

        analysis.markModified("report");

        await analysis.save();

        return res.status(200).json({
            success: true,
            message: "Deep scan completed",
            analysis,
        });

    } catch (error) {

        console.log(
            "Deep scan error:",
            error
        );

        return res.status(500).json({
            success: false,
            message:
                error.message ||
                "Deep scan failed",
        });
    }
};