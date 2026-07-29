import Product from "../models/Product.js";

export const createProduct = async (req, res) => {
    try {



        const { title, description, price } = req.body;

        if (!title || !description || !price) {
            return res.status(400).json({
                success: false,
                message: "Please fill all fields",
            });
        }

        const image = req.file.filename;
        const product = await Product.create({
            seller: req.user._id,
            title,
            description,
            price,
            image,
        });

        return res.status(201).json({
            success: true,
            message: "Product Created Successfully",
            product,
        });

    } catch (error) {
        res.status(500).json({
            success: false,
            message: error.message,
        });
    }
};