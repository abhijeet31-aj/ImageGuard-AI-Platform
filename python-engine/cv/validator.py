def validate_image(image, image_info, content_type):
    """
    Validate uploaded image before analysis.
    """

    if image is None:
        return False, "Image could not be decoded."

    if image_info["width"] <= 0:
        return False, "Invalid image width."

    if image_info["height"] <= 0:
        return False, "Invalid image height."

    if image_info["channels"] != 3:
        return False, "Only RGB images are supported."

    allowed_types = [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ]

    if content_type.lower() not in allowed_types:
        return False, "Unsupported image format."

    return True, "Valid Image"