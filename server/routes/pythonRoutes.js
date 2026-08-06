import express from "express";
import upload from "../middleware/uploadMiddleware.js";
import {
    checkPythonEngine,
    sendImageToPython
} from "../services/pythonService.js";

const router = express.Router();

// Health Check
router.get("/health", async (req, res) => {
    try {

        const result = await checkPythonEngine();

        res.json(result);

    } catch (error) {

        res.status(500).json({
            success: false,
            message: error.message
        });

    }
});

// Temporary Image Transfer Test
router.post(
    "/test-upload",
    upload.single("image"),
    async (req, res) => {

        try {

            if (!req.file) {
                return res.status(400).json({
                    success: false,
                    message: "No image uploaded"
                });
            }

            const result = await sendImageToPython(req.file.path, req.body.category);

            res.json(result);

        } catch (error) {

            console.log(error);

            res.status(500).json({
                success: false,
                message: error.message
            });

        }

    }
);

export default router;