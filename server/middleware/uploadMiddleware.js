import multer from "multer";
import path from "path";

const storage = multer.diskStorage({
    destination: "uploads/",

    filename: (req, file, cb) => {
        const extension = path.extname(file.originalname).toLowerCase();
        const uniqueName = Date.now() + extension;

        cb(null, uniqueName);
    },
});

// Maximum upload size in MB (override with the MAX_UPLOAD_MB env variable).
const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB) || 25;

const allowedExtensions = [
    ".jpg",
    ".jpeg",
    ".png",
];

const allowedMimeTypes = [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "application/octet-stream",
];

const fileFilter = (req, file, cb) => {

    const extension = path.extname(file.originalname).toLowerCase();

    const extensionValid = allowedExtensions.includes(extension);
    const mimeValid = allowedMimeTypes.includes(file.mimetype);

    if (extensionValid && mimeValid) {
        cb(null, true);
    } else {
        cb(new Error("Only JPG, JPEG and PNG images are allowed."));
    }
};

const upload = multer({
    storage: storage,
    fileFilter: fileFilter,
    limits: {
        fileSize: MAX_UPLOAD_MB * 1024 * 1024,
    },
});

export default upload;
