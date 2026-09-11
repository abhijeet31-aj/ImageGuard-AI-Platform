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
# AI DETECTION
# --------------------------------------------------

def detect_ai_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    # Apply training-compatible preprocessing
    image_tensor = image_transform(
        image
    ).unsqueeze(0).to(device)


    # Model inference
    with torch.no_grad():

        outputs = model(
            pixel_values=image_tensor
        )

        logit = outputs.logits.squeeze().item()


    # Binary classifier:
    # REAL = 0
    # FAKE/AI = 1

    ai_score = torch.sigmoid(
        torch.tensor(logit)
    ).item()


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