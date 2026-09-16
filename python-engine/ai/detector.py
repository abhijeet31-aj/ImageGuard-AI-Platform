from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification


MODEL_DIR = Path(r"D:\ImageGuardML\trained_model")

THRESHOLD = 0.35

device = torch.device("cpu")

print("Loading ImageGuardML model...")
print("Model path:", MODEL_DIR)
print("Device:", device)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = AutoModelForImageClassification.from_pretrained(
    MODEL_DIR
)

model = model.to(device)
model.eval()

print("ImageGuardML model loaded.")


# --------------------------------------------------
# PREPROCESSING
# SAME AS COMMUNITY FORENSICS VALIDATION/TEST
# --------------------------------------------------

image_transform = transforms.Compose([

    transforms.Resize(440),

    transforms.CenterCrop(384),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

])


# --------------------------------------------------
# CORE SCORING HELPER (shared by whole-image + tiled detection)
# --------------------------------------------------

def _score_pil_image(pil_image):
    """
    Run the model on a single already-loaded PIL image and return
    the raw AI-generation probability (sigmoid of the logit).

    Used by both detect_ai_image (whole image) and detect_ai_tiled
    (Phase 3 patch pipeline) so both paths always share the exact
    same preprocessing + model call.
    """

    image_tensor = image_transform(
        pil_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(
            pixel_values=image_tensor
        )

        logit = outputs.logits.squeeze().item()

    return torch.sigmoid(
        torch.tensor(logit)
    ).item()


# --------------------------------------------------
# AI DETECTION
# --------------------------------------------------

def detect_ai_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    ai_score = _score_pil_image(image)


    # Prediction
    prediction = (

        "AI Generated"

        if ai_score >= THRESHOLD

        else "Real"

    )


    # Confidence
    confidence = (

        ai_score

        if prediction == "AI Generated"

        else 1 - ai_score

    )


    return {

        "prediction":
            prediction,

        "ai_probability":
            round(ai_score, 4),

        "confidence":
            round(confidence, 4),

        "threshold":
            THRESHOLD

    }


# --------------------------------------------------
# TILED AI DETECTION — Phase 3 (Deep Scan, on-demand only)
# --------------------------------------------------
# Splits the image into overlapping tiles and runs the SAME model
# on each tile independently, to catch a partial AI edit that would
# be diluted/missed when scoring the whole image as one unit.
#
# IMPORTANT CAVEAT: this model was trained to look at Resize(440) +
# CenterCrop(384) of a full photo, relying on global semantic context.
# A small tile has to be upscaled to that size, which loses detail
# and gives a less reliable per-tile score than the whole-image score.
# So tile results should be read as a corroborating signal, not an
# independent ground truth — this is exactly why it's an optional
# "Deep Scan", not part of the default fast analysis.

TILE_HIGH_THRESHOLD = 0.60
TILE_LOW_THRESHOLD = 0.35
HETEROGENEITY_THRESHOLD = 0.22


def detect_ai_tiled(image_path, grid_target=4, overlap=0.5):
    """
    Run tiled AI-generation detection over the image.

    grid_target roughly controls how many tiles fit across the
    shorter image dimension (adaptive tile size), so a small image
    doesn't get sliced into dozens of tiny useless tiles and a huge
    image doesn't get an unbounded number of tiles.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    width, height = image.size

    tile_size = max(96, min(width, height) // grid_target)

    step = max(1, int(tile_size * (1 - overlap)))

    tiles = []

    for top in range(0, height, step):

        for left in range(0, width, step):

            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)

            # Skip degenerate slivers at the image edge
            if (right - left) < tile_size * 0.5:
                continue

            if (bottom - top) < tile_size * 0.5:
                continue

            crop = image.crop((left, top, right, bottom))

            score = _score_pil_image(crop)

            tiles.append({
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
                "score": round(score, 4),
            })

            # Whole row/column already covered exactly at the edge
            if right == width:
                break

        if bottom == height:
            break

    if not tiles:
        # Image too small to tile meaningfully — fall back to a
        # single whole-image score.
        whole_score = _score_pil_image(image)

        return {
            "tileCount": 0,
            "meanScore": round(whole_score, 4),
            "stdScore": 0.0,
            "tileVerdict": "Inconclusive (image too small to tile)",
            "suspiciousRegions": [],
        }

    scores = [tile["score"] for tile in tiles]

    mean_score = float(sum(scores) / len(scores))

    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

    std_score = float(variance ** 0.5)

    suspicious_regions = [
        tile for tile in tiles
        if tile["score"] >= TILE_HIGH_THRESHOLD
    ]

    max_score = max(scores)
    min_score = min(scores)

    # --------------------------------------------------
    # Tile verdict — documented heuristic (see roadmap Phase 3/4
    # for why this isn't a trained classifier yet)
    # --------------------------------------------------

    if (
        std_score >= HETEROGENEITY_THRESHOLD
        and max_score >= TILE_HIGH_THRESHOLD
        and min_score <= TILE_LOW_THRESHOLD
    ):
        tile_verdict = "Partially AI-Edited"

    elif mean_score >= TILE_HIGH_THRESHOLD:
        tile_verdict = "Consistent with Fully AI-Generated"

    elif mean_score <= TILE_LOW_THRESHOLD:
        tile_verdict = "Consistent with Authentic"

    else:
        tile_verdict = "Inconclusive"

    return {
        "tileCount": len(tiles),
        "meanScore": round(mean_score, 4),
        "stdScore": round(std_score, 4),
        "tileVerdict": tile_verdict,
        "suspiciousRegions": suspicious_regions,
    }