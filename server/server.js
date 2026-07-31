import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import connectDB from "./config/db.js";
import authRoutes from "./routes/authRoutes.js";
import protect from "./middleware/authMiddleware.js";
import imageRoutes from "./routes/imageRoutes.js";
import pythonRoutes from "./routes/pythonRoutes.js";

dotenv.config();

connectDB();

const app = express();

app.use(cors());
app.use(express.json());

app.use("/uploads", express.static("uploads"));

app.use("/api/auth", authRoutes);
app.use("/api/images", imageRoutes);

app.use("/api/python", pythonRoutes);

app.get("/", (req, res) => {
    res.send("ImageGuard API Running...");
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

app.get("/api/test", protect, (req, res) => {
    res.status(200).json({
        success: true,
        message: "Protected Route Accessed Successfully",
        user: req.user,
    });
});