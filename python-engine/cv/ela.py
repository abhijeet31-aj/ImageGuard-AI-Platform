from io import BytesIO

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def generate_difference_image(image_bytes, quality=90):
    """
    Generate RAW ELA difference image.
    No brightness enhancement is applied here.

    Returns:
        PIL.Image
    """

    original = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    temp_buffer = BytesIO()

    original.save(
        temp_buffer,
        "JPEG",
        quality=quality
    )

    temp_buffer.seek(0)

    compressed = Image.open(
        temp_buffer
    ).convert("RGB")

    difference_image = ImageChops.difference(
        original,
        compressed
    )

    return difference_image


def calculate_ela_metrics(difference_image):
    """
    Calculate metrics from RAW difference image.

    Returns:
        dict
    """

    ela_array = np.array(difference_image)

    mean_difference = float(
        np.mean(ela_array)
    )

    max_difference = int(
        np.max(ela_array)
    )

    standard_deviation = float(
        np.std(ela_array)
    )

    return {
        "meanDifference": round(mean_difference, 2),
        "maxDifference": max_difference,
        "standardDeviation": round(standard_deviation, 2)
    }


def create_ela_preview(difference_image):
    """
    Create a brightened ELA image
    only for visualization.

    Returns:
        PIL.Image
    """

    extrema = difference_image.getextrema()

    max_difference = max(
        channel[1]
        for channel in extrema
    )

    if max_difference == 0:
        max_difference = 1

    scale = 255.0 / max_difference

    preview = ImageEnhance.Brightness(
        difference_image
    ).enhance(scale)

    return preview