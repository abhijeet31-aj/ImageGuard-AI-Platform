import express from "express";
import protect from "../middleware/authMiddleware.js";
import upload from "../middleware/uploadMiddleware.js";
import {
    uploadImage,
    getAnalysisHistory,
    getAnalysisById
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
export default router;