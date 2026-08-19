from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


# Fine-tuned ImageGuardML model
MODEL_DIR = Path(r"D:\ImageGuardML\trained_model")

THRESHOLD = 0.35


# CPU because ImageGuard main project currently has CPU-only PyTorch
device = torch.device("cpu")

print("Loading ImageGuardML model...")
print("Model path:", MODEL_DIR)
print("Device:", device)


processor = AutoImageProcessor.from_pretrained(
    MODEL_DIR
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_DIR
)

model = model.to(device)
model.eval()

print("ImageGuardML model loaded.")


def detect_ai_image(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        logit = outputs.logits.squeeze().item()

        ai_score = torch.sigmoid(
            torch.tensor(logit)
        ).item()

    prediction = (
        "AI Generated"
        if ai_score >= THRESHOLD
        else "Real"
    )

    confidence = (
        ai_score
        if prediction == "AI Generated"
        else 1 - ai_score
    )

    return {
        "prediction": prediction,
        "ai_probability": round(ai_score, 4),
        "confidence": round(confidence, 4),
        "threshold": THRESHOLD
    }