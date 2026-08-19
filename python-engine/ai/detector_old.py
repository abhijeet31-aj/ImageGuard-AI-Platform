from transformers import pipeline
from PIL import Image


MODEL_NAME = "capcheck/ai-image-detection"


detector = pipeline(
    "image-classification",
    model=MODEL_NAME
)


def detect_ai_image(image_path):

    image = Image.open(image_path).convert("RGB")

    results = detector(image)

    return results