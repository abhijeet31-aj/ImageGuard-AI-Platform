import multer from "multer";
import path from "path";

const storage = multer.diskStorage({
    destination: "uploads/",
    filename: (req, file, cb) => {
        const uniqueName = Date.now() + path.extname(file.originalname);
        cb(null, uniqueName);
    },
});

const fileFilter = (req, file, cb) => {
    const allowedTypes = /jpg|jpeg|png/;

    const isValid =
        allowedTypes.test(path.extname(file.originalname).toLowerCase()) &&
        allowedTypes.test(file.mimetype);

    if (isValid) {
        cb(null, true);
    } else {
        cb(new Error("Only JPG, JPEG and PNG images are allowed."));
    }
};

const upload = multer({
    storage,
    fileFilter,
});

export default upload;