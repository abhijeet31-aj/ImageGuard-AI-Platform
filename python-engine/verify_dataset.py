"""
Phase 4 — Dataset verification.

Run this LOCALLY (on your machine, inside python-engine/) after
downloading and placing your datasets, BEFORE running
extract_features.py. It checks the things that usually go wrong
with large downloaded datasets:

    - correct folder structure
    - each class folder actually has images in it
    - every file can actually be opened (catches partial/corrupted
      downloads, wrong file extensions, zero-byte files)
    - a summary of formats + resolution range per class, so you can
      eyeball whether the download looks right

Usage:
    python verify_dataset.py
"""

from pathlib import Path
from collections import Counter

from PIL import Image, UnidentifiedImageError

from phase4_common import DATASET_DIR, CLASS_FOLDERS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_class_folder(folder_name, class_label):

    folder = DATASET_DIR / folder_name

    print(f"\n{'-' * 60}")
    print(f"{class_label}  ({folder})")
    print("-" * 60)

    if not folder.exists():
        print("  MISSING — this folder does not exist yet.")
        return {"found": False, "count": 0, "corrupt": 0}

    image_paths = find_images(folder)

    if not image_paths:
        print("  EMPTY — folder exists but no images found inside it.")
        print("  (Check you didn't leave everything inside a nested zip-extracted subfolder.)")
        return {"found": True, "count": 0, "corrupt": 0}

    format_counter = Counter()
    widths = []
    heights = []
    corrupt_files = []

    for path in image_paths:

        try:

            with Image.open(path) as img:
                img.verify()  # cheap corruption check

            with Image.open(path) as img:
                format_counter[img.format] += 1
                widths.append(img.width)
                heights.append(img.height)

        except (UnidentifiedImageError, OSError) as error:
            corrupt_files.append((path, str(error)))

    print(f"  Total images found : {len(image_paths)}")
    print(f"  Readable            : {len(image_paths) - len(corrupt_files)}")
    print(f"  Corrupt/unreadable  : {len(corrupt_files)}")

    if widths:
        print(f"  Resolution range    : {min(widths)}x{min(heights)}  to  {max(widths)}x{max(heights)}")

    if format_counter:
        formats_str = ", ".join(f"{fmt}={count}" for fmt, count in format_counter.items())
        print(f"  Formats             : {formats_str}")

    if corrupt_files:
        print("\n  First few corrupt files:")
        for path, error in corrupt_files[:5]:
            print(f"    - {path.relative_to(DATASET_DIR)}  ({error})")

        if len(corrupt_files) > 5:
            print(f"    ...and {len(corrupt_files) - 5} more")

    return {
        "found": True,
        "count": len(image_paths),
        "corrupt": len(corrupt_files),
    }


def main():

    print("=" * 60)
    print("PHASE 4 — DATASET VERIFICATION")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print(f"\nERROR: {DATASET_DIR} does not exist at all.")
        print("Create it and add the class subfolders first — see phase4_common.py docstring.")
        return

    summary = {}

    for folder_name, class_label in CLASS_FOLDERS.items():
        summary[class_label] = check_class_folder(folder_name, class_label)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    ready_for_extraction = True

    for class_label, result in summary.items():

        if not result["found"]:
            print(f"  {class_label:22s} NOT SET UP")
            ready_for_extraction = False

        elif result["count"] == 0:
            print(f"  {class_label:22s} EMPTY")
            ready_for_extraction = False

        else:
            usable = result["count"] - result["corrupt"]
            print(f"  {class_label:22s} {usable} usable image(s)"
                + (f"  ({result['corrupt']} corrupt, will be skipped)" if result["corrupt"] else ""))

    print()

    if ready_for_extraction:
        print("All class folders have usable images. You can run extract_features.py now.")
    else:
        print("Fix the issues above before running extract_features.py.")

    print("=" * 60)


if __name__ == "__main__":
    main()