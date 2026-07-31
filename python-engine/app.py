from fastapi import FastAPI, UploadFile, File

app = FastAPI(
    title="ImageGuard Python Engine",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "success": True,
        "message": "ImageGuard Python Engine Running"
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy"
    }


@app.post("/analyze-image")
async def analyze_image(image: UploadFile = File(...)):

    return {
        "success": True,
        "filename": image.filename,
        "contentType": image.content_type,
        "message": "Image received successfully"
    }