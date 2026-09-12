"""
Manipulation Detector — Phase 1

Purpose:
    detect_ai_image() (Community Forensics ViT) only answers
    "is this image fully AI-generated, or does it look like a real
    camera photo?". A real photo that has been spliced, copy-moved,
    or locally edited still looks like a real photo to that model,
    since its underlying pixels genuinely came from a camera.

    This module is a SEPARATE, classical-CV specialist that looks for
    signs of traditional editing/tampering instead. It does not use
    any trained model — no dataset or GPU is required for this phase.

Signals used:
    1. Noise inconsistency   — real camera sensor noise is roughly
                                uniform across an image. A block whose
                                noise differs sharply from the rest was
                                likely pasted in from elsewhere.
    2. ELA inconsistency     — a spliced/re-saved region has a different
                                double-compression response than the
                                rest of the image.
    3. Copy-move detection   — ORB keypoint self-matching to find
                                duplicated regions (clone-stamp editing).

Important limitation (report this honestly to the user):
    Classical CV forensic signals can false-positive on heavy JPEG
    compression, low-light sensor noise, legitimate HDR/panorama
    stitching, and some phone-camera computational photography
    pipelines. The thresholds below are a reasonable first-pass
    heuristic, NOT calibrated against labelled data — that calibration
    is planned for Phase 4 once labelled manipulated/authentic data is
    available. Until then, "Manipulated" should be read as "this image
    shows editing-consistent signals", not as a certainty claim, and
    borderline cases are intentionally routed to "Needs Review".
"""

import cv2
import numpy as np


BLOCK_SIZE = 64

# Minimum absolute noise/ELA level (0-255 scale) below which we don't
# trust a relative coefficient-of-variation number at all. Plain
# backgrounds (product photos, studio shots) can have near-zero noise
# in most blocks — a tiny absolute difference (e.g. 0 vs 6) there
# produces a huge RELATIVE ratio despite being visually and
# forensically meaningless. This floor prevents that from being
# mistaken for manipulation.
NOISE_ABSOLUTE_FLOOR = 1.0
ELA_ABSOLUTE_FLOOR = 1.0

# Heuristic signal weights, sanity-checked against the 10 labelled
# sample images already in test_images/ (original1-5, edited1-5).
# NOTE: this is a 10-image sanity check, NOT a proper held-out
# evaluation — treat these weights as a reasonable starting point,
# to be replaced by a properly trained meta-fusion model in Phase 2/4
# once a real labelled dataset is evaluated (see roadmap Phase 4).
# On that 10-image sample: noise and ELA inconsistency separated
# originals (combined score ~0.19-0.48) from edited (~0.53-0.77)
# cleanly; copy-move did not fire distinctively on this sample (the
# edits present were not copy-move style), so it is kept as a
# lower-weight contributor rather than dropped, since it should still
# help on clone-stamp-style edits.
NOISE_WEIGHT = 0.50
ELA_WEIGHT = 0.40
COPY_MOVE_WEIGHT = 0.10

MANIPULATED_THRESHOLD = 0.55
AUTHENTIC_THRESHOLD = 0.45


# ============================================================
# 1. NOISE INCONSISTENCY
# ============================================================

def _compute_block_texture_mask(gray, block_size=BLOCK_SIZE):
    """
    Split the image into blocks and mark which ones are "smooth"
    (low local texture/edge content) vs "detailed" (high texture).

    This matters because a real, untouched photo naturally has very
    different local noise/ELA response between a detailed region
    (grass, foliage, fabric) and a smooth region (sky, wall, skin) —
    that difference is just image CONTENT, not tampering. Comparing
    noise/ELA only across the smooth blocks avoids mistaking texture
    variation for manipulation, which is standard practice in noise
    forensics (PRNU-style analysis relies on flat regions for the
    same reason).

    Returns:
        positions      — list of (y, x) top-left corners of each block
        textures       — texture (Laplacian variance) score per block
        smooth_mask    — boolean array, True for the smoother half
                          of the image's blocks
    """

    height, width = gray.shape

    positions = []
    textures = []

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):

            block = gray[y:y + block_size, x:x + block_size]

            if block.size == 0:
                continue

            positions.append((y, x))
            textures.append(float(cv2.Laplacian(block, cv2.CV_64F).var()))

    textures = np.array(textures)

    if len(textures) == 0:
        return positions, textures, np.array([], dtype=bool)

    # Adaptive per-image threshold (median split) rather than a fixed
    # number — this way it adapts whether the photo is a busy outdoor
    # scene or a plain studio product shot.
    median_texture = float(np.median(textures))

    smooth_mask = textures <= median_texture

    return positions, textures, smooth_mask


def analyze_noise_consistency(image_cv, positions, smooth_mask, block_size=BLOCK_SIZE):
    """
    Measure how much local noise level varies across the SMOOTH
    blocks of the image only (see _compute_block_texture_mask for why).

    A high coefficient of variation (std / mean) among smooth-block
    noise levels suggests some flat-looking region has different
    noise characteristics than the rest — consistent with a pasted
    or heavily re-processed region.
    """

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    noise_map = cv2.absdiff(gray, blurred)

    all_noise_values = []

    for (y, x) in positions:
        block = noise_map[y:y + block_size, x:x + block_size]
        all_noise_values.append(float(np.std(block)) if block.size else 0.0)

    all_noise_values = np.array(all_noise_values)

    smooth_noise_values = all_noise_values[smooth_mask] if len(smooth_mask) else all_noise_values

    # Perfectly flat/clean blocks (near-zero noise) are common in
    # studio/product photography backgrounds and carry no real signal
    # to compare — including them turns a relative (std / mean)
    # comparison unstable, since dividing by a near-zero mean makes
    # any small absolute difference look huge. So we compare
    # variability only among smooth blocks that have SOME measurable
    # noise to begin with.
    measurable_noise_values = smooth_noise_values[smooth_noise_values > 0.15]

    if len(measurable_noise_values) < 4:
        # Whole image is essentially noise-free in its flat regions
        # (e.g. a clean product photo) — nothing meaningful to compare.
        return {
            "inconsistencyScore": 0.0,
            "outlierBlockCount": 0,
            "blockCount": int(len(measurable_noise_values)),
        }

    mean_noise = float(np.mean(measurable_noise_values))
    std_noise = float(np.std(measurable_noise_values))

    coefficient_of_variation = std_noise / max(mean_noise, NOISE_ABSOLUTE_FLOOR)

    median_noise = float(np.median(measurable_noise_values))

    mad = float(np.median(np.abs(measurable_noise_values - median_noise)))

    outlier_threshold = max(
        median_noise + (4 * mad if mad > 1e-6 else median_noise),
        NOISE_ABSOLUTE_FLOOR,
    )

    outlier_block_count = int(
        np.sum(measurable_noise_values > outlier_threshold)
    )

    inconsistency_score = min(coefficient_of_variation / 1.2, 1.0)

    return {
        "inconsistencyScore": round(inconsistency_score, 4),
        "outlierBlockCount": outlier_block_count,
        "blockCount": int(len(measurable_noise_values)),
    }


# ============================================================
# 2. ELA REGIONAL INCONSISTENCY
# ============================================================

def analyze_ela_consistency(difference_image, positions, smooth_mask, block_size=BLOCK_SIZE):
    """
    Take the raw ELA difference image (PIL.Image, already computed
    once in app.py — not recomputed here) and check whether the
    compression-error response varies sharply across the image's
    SMOOTH blocks (same reasoning as analyze_noise_consistency —
    edges/detailed regions legitimately produce a stronger ELA
    response even in an untouched photo, so comparing those would
    false-positive on ordinary detailed real photos).

    A spliced region that was compressed at a different time/quality
    than the rest of the image shows a different ELA response even
    among otherwise-smooth areas.
    """

    ela_array = np.array(difference_image.convert("L"))

    all_block_means = []

    for (y, x) in positions:
        block = ela_array[y:y + block_size, x:x + block_size]
        all_block_means.append(float(np.mean(block)) if block.size else 0.0)

    all_block_means = np.array(all_block_means)

    smooth_means = all_block_means[smooth_mask] if len(smooth_mask) else all_block_means

    # Same reasoning as noise: exclude blocks with essentially zero
    # ELA response (perfectly flat, cleanly-compressed regions) from
    # the relative comparison, so a near-zero mean doesn't turn tiny
    # absolute differences into a huge ratio.
    measurable_means = smooth_means[smooth_means > 0.15]

    if len(measurable_means) < 4:
        return {
            "inconsistencyScore": 0.0,
            "outlierBlockCount": 0,
            "blockCount": int(len(measurable_means)),
        }

    mean_of_means = float(np.mean(measurable_means))
    std_of_means = float(np.std(measurable_means))

    coefficient_of_variation = std_of_means / max(mean_of_means, ELA_ABSOLUTE_FLOOR)

    median_val = float(np.median(measurable_means))

    mad = float(np.median(np.abs(measurable_means - median_val)))

    outlier_threshold = max(
        median_val + (4 * mad if mad > 1e-6 else median_val),
        ELA_ABSOLUTE_FLOOR,
    )

    outlier_block_count = int(np.sum(measurable_means > outlier_threshold))

    inconsistency_score = min(coefficient_of_variation / 1.2, 1.0)

    return {
        "inconsistencyScore": round(inconsistency_score, 4),
        "outlierBlockCount": outlier_block_count,
        "blockCount": int(len(measurable_means)),
    }


# ============================================================
# 3. COPY-MOVE DETECTION (ORB self-matching)
# ============================================================

def detect_copy_move(
    image_cv,
    min_keypoint_distance=40,
    offset_bucket_size=10,
):
    """
    Detect duplicated regions within the same image using ORB
    keypoint self-matching.

    A naive version of this (just counting any two distant keypoints
    that match) false-positives heavily on natural repetitive
    textures — grass, brick, water, foliage all produce lots of
    self-similar keypoints that are NOT copy-move editing.

    Real copy-move editing pastes one region to another location as
    a rigid block, so genuine duplicated keypoints all share roughly
    the SAME offset (dx, dy) between original and pasted copy.
    Natural texture self-similarity does not share a consistent
    offset. So instead of counting raw matches, we bucket matches by
    their offset vector and look at the size of the largest cluster.
    """

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=800)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 20:
        return {
            "matchCount": 0,
            "largestClusterSize": 0,
            "copyMoveScore": 0.0,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    raw_matches = matcher.knnMatch(descriptors, descriptors, k=3)

    offset_buckets = {}

    valid_match_count = 0

    for match_group in raw_matches:

        for match in match_group:

            # Skip a keypoint matching itself
            if match.queryIdx == match.trainIdx:
                continue

            # Only count each pair once (i->j, not both i->j and j->i)
            if match.queryIdx > match.trainIdx:
                continue

            pt1 = np.array(keypoints[match.queryIdx].pt)
            pt2 = np.array(keypoints[match.trainIdx].pt)

            spatial_distance = float(np.linalg.norm(pt1 - pt2))

            if not (
                match.distance < 40
                and spatial_distance > min_keypoint_distance
            ):
                continue

            valid_match_count += 1

            dx, dy = (pt2 - pt1)

            bucket_key = (
                round(dx / offset_bucket_size),
                round(dy / offset_bucket_size),
            )

            offset_buckets[bucket_key] = offset_buckets.get(bucket_key, 0) + 1

    largest_cluster_size = max(offset_buckets.values()) if offset_buckets else 0

    # A genuine pasted region produces several keypoint pairs sharing
    # the same offset. A cluster of 1-2 is likely coincidental texture
    # similarity; 4+ sharing an offset is a much stronger signal.
    copy_move_score = min(max(largest_cluster_size - 1, 0) / 6.0, 1.0)

    return {
        "matchCount": int(valid_match_count),
        "largestClusterSize": int(largest_cluster_size),
        "copyMoveScore": round(copy_move_score, 4),
    }


# ============================================================
# 4. COMBINE ALL SIGNALS
# ============================================================

def analyze_manipulation(image_cv, difference_image):
    """
    Run all classical manipulation signals and combine them into
    a single manipulationAnalysis result.

    This is a heuristic v1 combination (fixed weights), not a
    trained model. Phase 2 will replace the fixed weights with a
    proper meta-fusion model once labelled data is available.
    """

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    positions, textures, smooth_mask = _compute_block_texture_mask(gray)

    noise_result = analyze_noise_consistency(image_cv, positions, smooth_mask)

    ela_result = analyze_ela_consistency(difference_image, positions, smooth_mask)

    copy_move_result = detect_copy_move(image_cv)

    manipulation_probability = (
        NOISE_WEIGHT * noise_result["inconsistencyScore"]
        + ELA_WEIGHT * ela_result["inconsistencyScore"]
        + COPY_MOVE_WEIGHT * copy_move_result["copyMoveScore"]
    )

    manipulation_probability = round(
        float(min(max(manipulation_probability, 0.0), 1.0)), 4
    )

    # --------------------------------------------------------
    # Evidence — only report signals that actually fired
    # --------------------------------------------------------

    evidence = []

    if noise_result["inconsistencyScore"] > 0.5:
        evidence.append(
            f"Noise inconsistency detected across "
            f"{noise_result['outlierBlockCount']} image region(s)"
        )

    if ela_result["inconsistencyScore"] > 0.5:
        evidence.append(
            f"Uneven compression response (ELA) across "
            f"{ela_result['outlierBlockCount']} image region(s)"
        )

    if copy_move_result["copyMoveScore"] > 0.3:
        evidence.append(
            f"Possible duplicated region detected "
            f"({copy_move_result['matchCount']} matching keypoint pairs)"
        )

    if not evidence:
        evidence.append("No strong manipulation indicators detected")

    # --------------------------------------------------------
    # Prediction — margin-based, avoids overconfident guessing
    # --------------------------------------------------------

    if manipulation_probability >= MANIPULATED_THRESHOLD:
        prediction = "Manipulated"

    elif manipulation_probability <= AUTHENTIC_THRESHOLD:
        prediction = "Authentic"

    else:
        prediction = "Needs Review"

    # Confidence reflects distance from the uncertain middle band,
    # not a calibrated probability (that requires Phase 4 training).
    confidence = round(
        min(abs(manipulation_probability - 0.5) * 2.2, 1.0), 4
    )

    return {
        "prediction": prediction,
        "manipulationProbability": manipulation_probability,
        "confidence": confidence,
        "evidence": evidence,
        "signals": {
            "noise": noise_result,
            "ela": ela_result,
            "copyMove": copy_move_result,
        },
    }