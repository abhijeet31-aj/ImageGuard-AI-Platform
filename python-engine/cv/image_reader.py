import cv2
import numpy as np


def read_image(image_bytes):
    """
    Reads uploaded image bytes and converts them into
    an OpenCV image.

    Returns:
        success (bool)
        image (numpy.ndarray)
        image_info (dict)
        error (str | None)
    """

    try:

        # Convert bytes to NumPy array
        np_array = np.frombuffer(image_bytes, np.uint8)

        # Decode image using OpenCV
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return False, None, None, "Invalid image file"

        # Image Properties
        height, width, channels = image.shape

        aspect_ratio = round(width / height, 2)

        orientation = (
            "Landscape"
            if width > height
            else "Portrait"
            if height > width
            else "Square"
        )

        image_info = {
            "width": width,
            "height": height,
            "channels": channels,
            "aspectRatio": aspect_ratio,
            "orientation": orientation,
        }

        return True, image, image_info, None

    except Exception as e:
        return False, None, None, str(e)