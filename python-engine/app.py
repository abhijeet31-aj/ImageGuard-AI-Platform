from fastapi import FastAPI, UploadFile, File, Form
from cv.image_reader import read_image
from cv.validator import validate_image
from cv.metadata import extract_metadata
from cv.metadata import extract_metadata, analyze_metadata
from cv.ela import (
    generate_difference_image,
    calculate_ela_metrics,
    create_ela_preview
)
from utils.csv_logger import save_ela_metrics

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
async def analyze_image(
    image: UploadFile = File(...),
    category: str = Form(...) 
    ):

    image_bytes = await image.read()

    success, image_cv, image_info, error = read_image(image_bytes)

    if not success:
        return {
            "success": False,
            "message": error
        }

    valid, message = validate_image(
        image_cv,
        image_info,
        image.content_type
    )

    if not valid:
        return {
            "success": False,
            "message": message
        }

    metadata = extract_metadata(image_bytes)

    metadata_analysis = analyze_metadata(metadata)

    difference_image = generate_difference_image(
    image_bytes
    )
    
    ela_metrics = calculate_ela_metrics(
        difference_image
    )
    
    ela_preview = create_ela_preview(
        difference_image
    )
    
    ela_preview.save("ela_output.jpg")
    
    save_ela_metrics(
        filename=image.filename,
        category=category,
        mean_difference=ela_metrics["meanDifference"],
        max_difference=ela_metrics["maxDifference"],
        standard_deviation=ela_metrics["standardDeviation"]
    )

    return {
        "success": True,
        "filename": image.filename,
        "contentType": image.content_type,
        **image_info,
        "metadata": metadata,
        "metadataAnalysis": metadata_analysis,
        "elaMetrics": ela_metrics
}
    