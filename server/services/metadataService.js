import sharp from "sharp";
import fs from "fs";

export const extractMetadata = async (imagePath) => {
    try {

        // Extract image metadata
        const metadata = await sharp(imagePath).metadata();

        // Get file information
        const fileStats = fs.statSync(imagePath);

        return {
            width: metadata.width,
            height: metadata.height,
            format: metadata.format,
            fileSize: fileStats.size,
        };

    } catch (error) {
        console.log(error);
        throw new Error("Failed to extract metadata");
    }
};