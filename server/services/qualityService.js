import sharp from "sharp";

export const analyzeResolution = (metadata) => {

    const { width, height } = metadata;

    const totalPixels = width * height;

    const megapixels = Number((totalPixels / 1000000).toFixed(2));

    let status = "";
    let score = 0;

    if (totalPixels >= 8000000) {
        status = "Excellent";
        score = 100;
    }

    else if (totalPixels >= 2000000) {
        status = "Good";
        score = 80;
    }

    else if (totalPixels >= 1000000) {
        status = "Average";
        score = 60;
    }

    else {
        status = "Low";
        score = 40;
    }

    return {

        width,

        height,

        totalPixels,

        status,

        score

    };

};


export const analyzeBrightness = async (imagePath) => {

    // Resize image for faster processing
    const { data, info } = await sharp(imagePath)
        .resize(512, 512, {
            fit: "inside",
        })
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    let totalBrightness = 0;

    // RGB = 3 channels
    for (let i = 0; i < data.length; i += info.channels) {

        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        // Weighted Luminance Formula
        const brightness =
            (0.299 * r) +
            (0.587 * g) +
            (0.114 * b);

        totalBrightness += brightness;
    }

    const totalPixels = info.width * info.height;

    const averageBrightness = totalBrightness / totalPixels;

    let status = "";
    let score = 0;

    if (averageBrightness < 70) {

        status = "Too Dark";
        score = 40;

    } else if (averageBrightness <= 180) {

        status = "Normal";
        score = 100;

    } else {

        status = "Too Bright";
        score = 40;
    }

    return {

        averageBrightness: Number(
            averageBrightness.toFixed(2)
        ),

        status,

        score,
    };
};