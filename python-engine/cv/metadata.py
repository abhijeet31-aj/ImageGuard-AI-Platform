from io import BytesIO
from PIL import Image
from PIL.ExifTags import TAGS


def clean_exif_value(value):
    if isinstance(value, str):
        return value.replace("\x00", "").strip()
    return value

def extract_metadata(image_bytes):
    """
    Extract metadata directly from image bytes.

    Returns:
        dict
    """

    metadata = {
        "cameraMake": None,
        "cameraModel": None,
        "software": None,
        "dateTaken": None,
        "gpsAvailable": False,
        "imageWidth": None,
        "imageHeight": None
    }

    try:

        image = Image.open(BytesIO(image_bytes))

        metadata["imageWidth"] = image.width
        metadata["imageHeight"] = image.height

        exif_data = image.getexif()

        if not exif_data:
            return metadata

        exif = {}

        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            exif[tag] = value

        metadata["cameraMake"] = clean_exif_value(exif.get("Make"))
        metadata["cameraModel"] = clean_exif_value(exif.get("Model"))
        metadata["software"] = clean_exif_value(exif.get("Software"))
        metadata["dateTaken"] = clean_exif_value(exif.get("DateTime"))
        if "GPSInfo" in exif:
            metadata["gpsAvailable"] = True

        return metadata

    except Exception:
        return metadata
    
    
def analyze_metadata(metadata):
    """
    Analyze extracted metadata and generate
    integrity score and findings.
    """

    score = 50
    findings = []

    if metadata["cameraMake"]:
        score += 20
        findings.append("Camera manufacturer information available.")

    if metadata["cameraModel"]:
        score += 20
        findings.append("Camera model information available.")

    if metadata["dateTaken"]:
        score += 15
        findings.append("Capture date available.")

    if metadata["gpsAvailable"]:
        score += 5
        findings.append("GPS metadata available.")

    editing_software = [
        "Adobe",
        "Photoshop",
        "GIMP",
        "Canva",
        "Snapseed",
        "Lightroom",
        "PicsArt"
    ]

    software = metadata.get("software")

    if software:

        findings.append(f"Software detected: {software}")

        if any(app.lower() in software.lower() for app in editing_software):
            score -= 25
            findings.append("Editing software detected.")

    score = max(0, min(score, 100))

    if score >= 80:
        risk = "Low"
    elif score >= 60:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "integrityScore": score,
        "riskLevel": risk,
        "findings": findings
    }