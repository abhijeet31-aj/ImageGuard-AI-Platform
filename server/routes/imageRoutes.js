import express from "express";
import protect from "../middleware/authMiddleware.js";
import upload from "../middleware/uploadMiddleware.js";
import {
    uploadImage,
    getAnalysisHistory,
    getAnalysisById,
    deepScanImage
} from "../controllers/imageController.js";

const router = express.Router();


// Upload Image
router.post(
    "/upload",
    protect,
    upload.single("image"),
    uploadImage
);

// Analysis History
router.get(
    "/history",
    protect,
    getAnalysisHistory
);

// Single Analysis Report
router.get(
    "/:id",
    protect,
    getAnalysisById
);

// Deep Scan (Phase 3 — on-demand patch/tile pipeline)
router.post(
    "/:id/deep-scan",
    protect,
    deepScanImage
);

export default router;