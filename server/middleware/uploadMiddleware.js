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

const fileFilter = (req, file, cb) => {
    console.log("========== UPLOAD DEBUG ==========");
    console.log("Original name:", file.originalname);
    console.log("MIME type:", file.mimetype);
    console.log("Extension:", path.extname(file.originalname));
    console.log("==================================");

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

    const extension = path.extname(file.originalname).toLowerCase();

    const extensionValid = allowedExtensions.includes(extension);
    const mimeValid = allowedMimeTypes.includes(file.mimetype);

    console.log("Extension valid:", extensionValid);
    console.log("MIME valid:", mimeValid);

    if (extensionValid && mimeValid) {
        console.log("Final valid: true");
        cb(null, true);
    } else {
        console.log("Final valid: false");
        cb(new Error("Only JPG, JPEG and PNG images are allowed."));
    }
};

const upload = multer({
    storage: storage,
    fileFilter: fileFilter,
});

export default upload;