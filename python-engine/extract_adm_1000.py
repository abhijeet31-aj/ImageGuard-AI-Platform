from huggingface_hub import hf_hub_download
import pyarrow.ipc as ipc
from PIL import Image
from io import BytesIO
from pathlib import Path


# =========================
# SETTINGS
# =========================

REQUIRED_REAL = 500
REQUIRED_AI = 500

OUTPUT_DIR = Path(__file__).resolve().parent / "external-test" / "ADM"

REAL_DIR = OUTPUT_DIR / "REAL"
AI_DIR = OUTPUT_DIR / "AI"

REAL_DIR.mkdir(parents=True, exist_ok=True)
AI_DIR.mkdir(parents=True, exist_ok=True)


# ADM Arrow shards
SHARDS = [
    "data-00000-of-00003.arrow",
    "data-00001-of-00003.arrow",
    "data-00002-of-00003.arrow",
]


real_count = 0
ai_count = 0


print("=" * 60)
print("ADM EXTERNAL TEST DATASET EXTRACTION")
print("=" * 60)


for shard_name in SHARDS:

    if real_count >= REQUIRED_REAL and ai_count >= REQUIRED_AI:
        break

    print()
    print("Processing:", shard_name)

    path = hf_hub_download(
        repo_id="nebula/GenImage-arrow",
        filename=f"data/test/ADM/{shard_name}",
        repo_type="dataset"
    )

    with open(path, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()

    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()

    for image_bytes, label in zip(images, labels):

        # Label 0 = REAL
        if label == 0:

            if real_count >= REQUIRED_REAL:
                continue

            try:
                image = Image.open(BytesIO(image_bytes))
                image = image.convert("RGB")

                output_path = REAL_DIR / f"real_{real_count + 1:04d}.jpg"
                image.save(output_path, "JPEG", quality=95)

                real_count += 1

            except Exception as e:
                print("REAL image error:", e)

        # Label 1 = AI / FAKE
        elif label == 1:

            if ai_count >= REQUIRED_AI:
                continue

            try:
                image = Image.open(BytesIO(image_bytes))
                image = image.convert("RGB")

                output_path = AI_DIR / f"ai_{ai_count + 1:04d}.jpg"
                image.save(output_path, "JPEG", quality=95)

                ai_count += 1

            except Exception as e:
                print("AI image error:", e)


print()
print("=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print("REAL images:", real_count)
print("AI images:", ai_count)
print("TOTAL:", real_count + ai_count)

print()
print("Output directory:")
print(OUTPUT_DIR)

if real_count == REQUIRED_REAL and ai_count == REQUIRED_AI:
    print()
    print("SUCCESS: Balanced 1000-image dataset created.")
else:
    print()
    print("WARNING: Required number of images was not reached.")