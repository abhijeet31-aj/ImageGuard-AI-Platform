import ImageAnalysis from "../models/ImageAnalysis.js";

export const checkDuplicateImage = async (userId, imageHash) => {
    try {
        const existingImage = await ImageAnalysis.findOne({
            uploadedBy: userId,
            imageHash: imageHash,
        }).sort({ createdAt: -1 });

        return existingImage;
    } catch (error) {
        console.log(error);
        throw new Error("Duplicate check failed");
    }
};