import axios from "axios";

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