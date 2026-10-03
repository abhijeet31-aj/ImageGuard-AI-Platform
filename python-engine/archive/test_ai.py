from pathlib import Path
from ai.detector import detect_ai_image

TEST_FOLDER = Path("test_images")

extensions = {".jpg", ".jpeg", ".png", ".webp"}

for image_path in TEST_FOLDER.iterdir():

    if image_path.suffix.lower() not in extensions:
        continue

    try:
        result = detect_ai_image(str(image_path))

        print(f"\n{image_path.name}")
        print(result)

    except Exception as error:
        print(f"\n{image_path.name} -> ERROR: {error}")