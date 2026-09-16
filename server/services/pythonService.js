import axios from "axios";
import FormData from "form-data";
import fs from "fs";



const PYTHON_API = "http://127.0.0.1:8000";

export const checkPythonEngine = async () => {
    try {

        const response = await axios.get(
            `${PYTHON_API}/health`
        );

        return response.data;

    } catch (error) {

        throw new Error("Python Engine Not Running");

    }
};

export const sendImageToPython = async (imagePath, category) => {

    const formData = new FormData();

    formData.append(
        "image",
        fs.createReadStream(imagePath)
    );

    formData.append(
        "category",
        category
    )
    const response = await axios.post(

        `${PYTHON_API}/analyze-image`,

        formData,

        {
            headers: formData.getHeaders(),
        }

    );

    return response.data;

};


export const sendImageForDeepScan = async (imagePath, finalAnalysis) => {

    const formData = new FormData();

    formData.append(
        "image",
        fs.createReadStream(imagePath)
    );

    if (finalAnalysis) {

        formData.append(
            "final_analysis",
            JSON.stringify(finalAnalysis)
        );
    }

    const response = await axios.post(

        `${PYTHON_API}/deep-scan`,

        formData,

        {
            headers: formData.getHeaders(),
        }

    );

    return response.data;

};