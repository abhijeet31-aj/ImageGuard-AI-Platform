import io
import pandas as pd
from pathlib import Path
from PIL import Image

DATASET_DIR = Path("datasets")          # python-engine/datasets
df = pd.read_csv("phase4_dataset_features.csv")

def jpeg_quality_est(im):
    """Rough JPEG quality estimate from quantization table (None if not JPEG)."""
    q = getattr(im, "quantization", None)
    if not q:
        return None
    table = q.get(0)
    if not table:
        return None
    avg = sum(table) / len(table)
    # lower average quantizer value = higher quality
    return round(max(1, min(100, 100 - avg * 1.0)), 1)

rows = []
for fn, cls in zip(df["filename"], df["class_label"]):
    p = DATASET_DIR / fn.replace("\\", "/")
    try:
        size_bytes = p.stat().st_size
        with Image.open(p) as im:
            w, h = im.size
            fmt = im.format
            exif = im.getexif()
            rows.append(dict(
                filename=fn, class_label=cls, width=w, height=h,
                megapixels=round(w * h / 1e6, 3), format=fmt,
                file_bytes=size_bytes,
                bytes_per_px=round(size_bytes / (w * h), 3),
                jpeg_q_est=jpeg_quality_est(im),
                has_exif=bool(exif), exif_make=str(exif.get(271, ""))[:20],
            ))
    except Exception as e:
        rows.append(dict(filename=fn, class_label=cls, error=str(e)))

out = pd.DataFrame(rows)
out.to_csv("image_audit.csv", index=False)
print("done:", len(out), "rows; errors:", out.get("error", pd.Series()).notna().sum())
print(out.groupby("class_label")[["megapixels", "bytes_per_px", "jpeg_q_est"]].describe().T.to_string())
print(out.groupby("class_label").apply(lambda g: (g.megapixels >= 8).sum()).rename("images >= 8MP"))