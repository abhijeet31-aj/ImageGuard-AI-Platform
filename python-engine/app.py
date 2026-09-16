from pathlib import Path
import tempfile
import json

from fastapi import FastAPI, UploadFile, File, Form

from cv.image_reader import read_image
from cv.validator import validate_image
from cv.metadata import extract_metadata, analyze_metadata
from cv.ela import (
    generate_difference_image,
    calculate_ela_metrics,
    create_ela_preview
)
from cv.forgery import analyze_forgery
from cv.manipulation import analyze_manipulation

from ai.detector import detect_ai_image, detect_ai_tiled

from services.fusion import predict_fusion
from services.final_fusion import combine_final_analysis, refine_with_tile_analysis

from utils.csv_logger import save_ela_metrics
from utils.forgery_csv_logger import save_forgery_metrics


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

    # --------------------------------------------------
    # 1. READ IMAGE
    # --------------------------------------------------

    success, image_cv, image_info, error = read_image(
        image_bytes
    )

    if not success:
        return {
            "success": False,
            "message": error
        }


    # --------------------------------------------------
    # 2. VALIDATE IMAGE
    # --------------------------------------------------

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


    # --------------------------------------------------
    # 3. SAVE TEMPORARY IMAGE FOR AI MODEL
    # --------------------------------------------------

    suffix = Path(image.filename).suffix or ".jpg"

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:

            temp_file.write(image_bytes)
            temp_path = temp_file.name


        # --------------------------------------------------
        # 4. AI DETECTION - ImageGuardML
        # --------------------------------------------------

        ai_detection = detect_ai_image(
            temp_path
        )

    finally:

        if temp_path:
            Path(temp_path).unlink(
                missing_ok=True
            )


    # --------------------------------------------------
    # 5. METADATA ANALYSIS
    # --------------------------------------------------

    metadata = extract_metadata(
        image_bytes
    )

    metadata_analysis = analyze_metadata(
        metadata
    )


    # --------------------------------------------------
    # 6. ELA ANALYSIS
    # --------------------------------------------------

    difference_image = generate_difference_image(
        image_bytes
    )

    ela_metrics = calculate_ela_metrics(
        difference_image
    )

    ela_preview = create_ela_preview(
        difference_image
    )

    ela_preview.save(
        "ela_output.jpg"
    )


    save_ela_metrics(
        filename=image.filename,
        category=category,
        mean_difference=ela_metrics["meanDifference"],
        max_difference=ela_metrics["maxDifference"],
        standard_deviation=ela_metrics["standardDeviation"]
    )


    # --------------------------------------------------
    # 7. FORGERY ANALYSIS
    # --------------------------------------------------

    forgery_analysis = analyze_forgery(
        image_cv
    )


    save_forgery_metrics(
        filename=image.filename,
        category=category,
        edge_density=forgery_analysis["edgeDensity"],
        noise_mean=forgery_analysis["noiseMean"],
        noise_std=forgery_analysis["noiseStd"],
        blur_score=forgery_analysis["blurScore"],
        sharpness_score=forgery_analysis["sharpnessScore"],
        block_average=forgery_analysis["blockSharpnessAverage"],
        block_std=forgery_analysis["blockSharpnessStd"]
    )


    # --------------------------------------------------
    # 7b. MANIPULATION ANALYSIS (classical CV, Phase 1)
    # --------------------------------------------------
    # Separate from aiDetection/fusionAnalysis: this checks for
    # traditional editing/tampering signs (splice, copy-move,
    # regional inconsistency), which the AI-generation model does
    # not look for.

    manipulation_analysis = analyze_manipulation(
        image_cv,
        difference_image
    )


    # --------------------------------------------------
    # 8. FUSION ANALYSIS
    # --------------------------------------------------

    fusion_analysis = predict_fusion(
        ai_probability=ai_detection["ai_probability"],
        forensic_analysis={
            "meanDifference": ela_metrics["meanDifference"],
            "maxDifference": ela_metrics["maxDifference"],
            "standardDeviation": ela_metrics["standardDeviation"],
            "edgeDensity": forgery_analysis["edgeDensity"],
            "noiseMean": forgery_analysis["noiseMean"],
            "noiseStd": forgery_analysis["noiseStd"],
            "blurScore": forgery_analysis["blurScore"],
            "sharpnessScore": forgery_analysis["sharpnessScore"],
            "blockSharpnessAverage": forgery_analysis["blockSharpnessAverage"],
            "blockSharpnessStd": forgery_analysis["blockSharpnessStd"]
        }
    )


    # --------------------------------------------------
    # 8b. FINAL ANALYSIS (Phase 2 — rule-based combination)
    # --------------------------------------------------
    # Combines fusionAnalysis (AI-generation) + manipulationAnalysis
    # (Phase 1) into one final classification. See
    # services/final_fusion.py docstring for why this is a documented
    # rule, not a trained model, at this stage.

    final_analysis = combine_final_analysis(
        fusion_analysis,
        manipulation_analysis
    )


    # --------------------------------------------------
    # 9. FINAL RESPONSE
    # --------------------------------------------------

    return {
        "success": True,

        "filename": image.filename,

        "contentType": image.content_type,

        **image_info,

        "aiDetection": ai_detection,

        "metadata": metadata,

        "metadataAnalysis": metadata_analysis,

        "elaMetrics": ela_metrics,

        "forgeryAnalysis": forgery_analysis,

        "manipulationAnalysis": manipulation_analysis,

        "fusionAnalysis": fusion_analysis,

        "finalAnalysis": final_analysis
    }


# --------------------------------------------------
# DEEP SCAN — Phase 3 (on-demand patch/tile pipeline)
# --------------------------------------------------
# Separate, optional endpoint — NOT part of the default /analyze-image
# flow, since tiling runs the AI model many times over one image and
# is noticeably slower. The Node server calls this only when the user
# explicitly clicks "Deep Scan" on an existing result.

@app.post("/deep-scan")
async def deep_scan(
    image: UploadFile = File(...),
    final_analysis: str = Form(None)
):

    image_bytes = await image.read()

    suffix = Path(image.filename).suffix or ".jpg"

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:

            temp_file.write(image_bytes)
            temp_path = temp_file.name

        tile_analysis = detect_ai_tiled(
            temp_path
        )

    finally:

        if temp_path:
            Path(temp_path).unlink(
                missing_ok=True
            )


    refined_final_analysis = None

    if final_analysis:

        try:

            existing_final_analysis = json.loads(final_analysis)

            refined_final_analysis = refine_with_tile_analysis(
                existing_final_analysis,
                tile_analysis
            )

        except (json.JSONDecodeError, TypeError):

            refined_final_analysis = None


    return {
        "success": True,

        "tileAnalysis": tile_analysis,

        "suspiciousRegions": tile_analysis.get("suspiciousRegions", []),

        "finalAnalysis": refined_final_analysis
    }