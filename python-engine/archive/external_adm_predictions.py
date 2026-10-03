import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = Path(r"D:\ImageGuardML\trained_model")

REAL_DIR = BASE_DIR / "external-test" / "ADM" / "REAL"
AI_DIR = BASE_DIR / "external-test" / "ADM" / "AI"

OUTPUT_FILE = BASE_DIR / "external_adm_predictions.csv"


# ==========================================
# SETTINGS
# ==========================================

THRESHOLD = 0.35

device = torch.device("cpu")


# ==========================================
# LOAD MODEL
# ==========================================

print("=" * 60)
print("LOADING IMAGeGUARD AI MODEL")
print("=" * 60)

print("Model:", MODEL_DIR)
print("Device:", device)

model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()


# ==========================================
# IMAGE TRANSFORM
# ==========================================

image_transform = transforms.Compose([
    transforms.Resize(440),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    logit = outputs.logits.squeeze().item()

    ai_probability = torch.sigmoid(
        torch.tensor(logit)
    ).item()

    prediction = "AI" if ai_probability >= THRESHOLD else "REAL"

    return ai_probability, prediction


# ==========================================
# PROCESS DATASET
# ==========================================

results = []

total = 1000
processed = 0


print()
print("=" * 60)
print("STARTING EXTERNAL ADM TEST")
print("=" * 60)


# ------------------------------------------
# REAL IMAGES
# ------------------------------------------

for image_path in sorted(REAL_DIR.glob("*")):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    try:

        ai_probability, prediction = predict_image(image_path)

        results.append({
            "filename": image_path.name,
            "true_label": "REAL",
            "ai_probability": ai_probability,
            "prediction": prediction
        })

        processed += 1

        if processed % 50 == 0:
            print(
                f"Processed: {processed}/{total}"
            )

    except Exception as e:

        print(
            f"Error processing {image_path.name}: {e}"
        )


# ------------------------------------------
# AI IMAGES
# ------------------------------------------

for image_path in sorted(AI_DIR.glob("*")):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    try:

        ai_probability, prediction = predict_image(image_path)

        results.append({
            "filename": image_path.name,
            "true_label": "AI",
            "ai_probability": ai_probability,
            "prediction": prediction
        })

        processed += 1

        if processed % 50 == 0:
            print(
                f"Processed: {processed}/{total}"
            )

    except Exception as e:

        print(
            f"Error processing {image_path.name}: {e}"
        )


# ==========================================
# SAVE CSV
# ==========================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "true_label",
            "ai_probability",
            "prediction"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# SUMMARY
# ==========================================

print()
print("=" * 60)
print("EXTERNAL TEST COMPLETE")
print("=" * 60)

print("Total processed:", len(results))
print("Expected:", total)

print()
print("Prediction file:")
print(OUTPUT_FILE)