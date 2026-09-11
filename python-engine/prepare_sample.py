from pathlib import Path
import pyarrow.ipc as ipc
from PIL import Image
import io

ARROW_FILE = Path(
    r"C:\Users\ABHIJEET\.cache\huggingface\hub\datasets--nebula--GenImage-arrow\snapshots\3f4b9f921a673be09a93b335ed728cea0c6ecf33\data\test\ADM\data-00000-of-00003.arrow"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "external-test" / "sample"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("GENIMAGE SAMPLE EXTRACTION")
print("=" * 70)

with open(ARROW_FILE, "rb") as file:
    reader = ipc.open_stream(file)
    table = reader.read_all()

print("Total rows in shard:", table.num_rows)
print("Columns:", table.column_names)
print()


images = table.column("image")
labels = table.column("label")
paths = table.column("image_path")


real_count = 0
ai_count = 0
saved_count = 0

for i in range(table.num_rows):

    label = labels[i].as_py()

    # We only need 5 REAL + 5 AI
    if label == 0 and real_count >= 5:
        continue

    if label == 1 and ai_count >= 5:
        continue

    image_bytes = images[i].as_py()
    original_path = paths[i].as_py()

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()

        if label == 0:
            real_count += 1
            output_name = f"real_{real_count:02d}.jpg"

        else:
            ai_count += 1
            output_name = f"ai_{ai_count:02d}.jpg"

        output_path = OUTPUT_DIR / output_name

        # Save as JPEG for a consistent external test format
        image.convert("RGB").save(
            output_path,
            format="JPEG",
            quality=95
        )

        saved_count += 1

        print(
            f"{output_name} | "
            f"label={'REAL' if label == 0 else 'AI'} | "
            f"source={original_path}"
        )

    except Exception as error:
        print(f"ERROR at row {i}: {error}")

    if real_count >= 5 and ai_count >= 5:
        break


print()
print("=" * 70)
print("EXTRACTION SUMMARY")
print("=" * 70)
print("REAL images:", real_count)
print("AI images  :", ai_count)
print("Total      :", saved_count)
print("Output     :", OUTPUT_DIR)
print("=" * 70)