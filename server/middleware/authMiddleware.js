import jwt from "jsonwebtoken";
import User from "../models/User.js";

const protect = async (req, res, next) => {
    try {
        if (
            !req.headers.authorization ||
            !req.headers.authorization.startsWith("Bearer")
        ) {
            return res.status(401).json({
                success: false,
                message: "Not Authorized",
            });
        }

        const token = req.headers.authorization.split(" ")[1];

        const decoded = jwt.verify(token, process.env.JWT_SECRET);

        req.user = await User.findById(decoded.id).select("-password");

        next();
    } catch (error) {
        return res.status(401).json({
            success: false,
            message: "Token Invalid",
        });
    }
};

export default protect;