import express from "express";
import protect from "../middleware/authMiddleware.js";
import upload from "../middleware/uploadMiddleware.js";
import { uploadImage } from "../controllers/imageController.js";

const router = express.Router();

// Upload Image
router.post(
    "/upload",
    protect,
    upload.single("image"),
    uploadImage
);

export default router;