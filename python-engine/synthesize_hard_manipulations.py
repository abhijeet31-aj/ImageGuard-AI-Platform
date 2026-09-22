"""
Phase 4 — Synthesize "hard" (ELA-resistant) manipulated training
examples.

error_analysis.py found that the model's worst confusions
(Manipulated -> Authentic, 32 cases) are driven mainly by LOW
maxDifference/standardDeviation (ELA metrics) — i.e. the model has
almost no manipulated training examples that DON'T show a strong
double-compression signature, so it never learned to catch a
manipulation using other signals (noise/regional/copy-move) when ELA
stays quiet.

This script creates exactly that: genuine splice/copy-move edits,
saved ONCE as a fresh high-quality JPEG (no double-compression
history), directly targeting the diagnosed gap. This is not a
replacement for a real dataset like IMD2020 — it's a fast, immediate
supplement using material you already have.

Usage:
    python synthesize_hard_manipulations.py [count]
    (default count: 300)

Reads from:  datasets/authentic/  (as source material — untouched)
Writes to:   datasets/manipulated/synthetic_hard/   (new subfolder;
             extract_features.py already walks subfolders recursively,
             so no other changes needed before re-running it)
"""

import sys
import random
from pathlib import Path

import cv2
import numpy as np

from phase4_common import DATASET_DIR


SOURCE_DIR = DATASET_DIR / "authentic"
OUTPUT_DIR = DATASET_DIR / "manipulated" / "synthetic_hard"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Saved once, at high quality — no double-compression artefact for
# ELA to catch, matching the real "hard case" characteristic found.
JPEG_QUALITY = 95

MIN_PATCH_FRACTION = 0.08   # smallest patch: ~8% of image area
MAX_PATCH_FRACTION = 0.20   # largest patch: ~20% of image area


def find_images(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def random_patch_box(width, height, rng):

    fraction = rng.uniform(MIN_PATCH_FRACTION, MAX_PATCH_FRACTION)

    patch_w = int(width * (fraction ** 0.5))
    patch_h = int(height * (fraction ** 0.5))

    patch_w = max(16, min(patch_w, width - 1))
    patch_h = max(16, min(patch_h, height - 1))

    x = rng.randint(0, width - patch_w)
    y = rng.randint(0, height - patch_h)

    return x, y, patch_w, patch_h


def copy_move(image, rng):
    """
    Copy a patch from one location in the image, paste it somewhere
    else in the SAME image — classic copy-move forgery.
    """

    height, width = image.shape[:2]

    src_x, src_y, patch_w, patch_h = random_patch_box(width, height, rng)

    patch = image[src_y:src_y + patch_h, src_x:src_x + patch_w].copy()

    dst_x = rng.randint(0, width - patch_w)
    dst_y = rng.randint(0, height - patch_h)

    result = image.copy()
    result[dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = patch

    return result


def splice(image, donor_image, rng):
    """
    Paste a patch from a DIFFERENT (donor) image into this one —
    classic splicing forgery.
    """

    height, width = image.shape[:2]
    donor_height, donor_width = donor_image.shape[:2]

    patch_w = min(width, donor_width) // 3
    patch_h = min(height, donor_height) // 3

    patch_w = max(16, patch_w)
    patch_h = max(16, patch_h)

    src_x = rng.randint(0, max(1, donor_width - patch_w))
    src_y = rng.randint(0, max(1, donor_height - patch_h))

    patch = donor_image[src_y:src_y + patch_h, src_x:src_x + patch_w]

    dst_x = rng.randint(0, max(1, width - patch_w))
    dst_y = rng.randint(0, max(1, height - patch_h))

    result = image.copy()

    paste_h = min(patch_h, height - dst_y)
    paste_w = min(patch_w, width - dst_x)

    result[dst_y:dst_y + paste_h, dst_x:dst_x + paste_w] = patch[:paste_h, :paste_w]

    return result


def main():

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    if not SOURCE_DIR.exists():
        print(f"ERROR: {SOURCE_DIR} does not exist.")
        sys.exit(1)

    source_images = find_images(SOURCE_DIR)

    if len(source_images) < 2:
        print("Need at least 2 authentic images to splice from.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    written = 0
    failed = 0

    for index in range(count):

        try:

            image_path = rng.choice(source_images)
            image = cv2.imread(str(image_path))

            if image is None:
                failed += 1
                continue

            # Alternate between copy-move and splice for variety
            if index % 2 == 0:

                result = copy_move(image, rng)
                tag = "copymove"

            else:

                donor_path = rng.choice(source_images)
                donor_image = cv2.imread(str(donor_path))

                if donor_image is None:
                    failed += 1
                    continue

                result = splice(image, donor_image, rng)
                tag = "splice"

            output_path = OUTPUT_DIR / f"synthetic_{tag}_{index:04d}.jpg"

            cv2.imwrite(
                str(output_path),
                result,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            written += 1

        except Exception as error:

            failed += 1
            print(f"  Failed on image {index}: {error}")

        if (index + 1) % 50 == 0:
            print(f"  ...{index + 1}/{count}")

    print(f"\nWritten: {written}   Failed: {failed}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(
        "\nThese are now inside datasets/manipulated/synthetic_hard/ — "
        "extract_features.py will pick them up automatically on its "
        "next run (it walks subfolders recursively). Delete "
        "phase4_dataset_features.csv first since the manipulated "
        "class's total count changed."
    )


if __name__ == "__main__":
    main()