import express from "express";
import { checkPythonEngine } from "../services/pythonService.js";

const router = express.Router();

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

export default router;