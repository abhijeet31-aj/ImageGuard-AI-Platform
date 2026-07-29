
import express from "express";
import protect from "../middleware/authMiddleware.js";
import upload from "../middleware/uploadMiddleware.js";
import { createProduct } from "../controllers/productController.js";



const router = express.Router();

router.post(
    "/create",
    protect,
    upload.single("image"),
    createProduct
);


export default router;