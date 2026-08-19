import cv2
import numpy as np


def detect_edges(image_cv):
    """
    Detect image edges using Canny Edge Detection.
    """

    gray = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_pixels = np.count_nonzero(edges)

    total_pixels = edges.shape[0] * edges.shape[1]

    edge_density = (
        edge_pixels / total_pixels
    ) * 100

    return {
        "edgeImage": edges,
        "edgeDensity": round(
            edge_density,
            2
        )
    }
    
    
def analyze_noise(image_cv):
    """
    Analyze image noise using Gaussian Blur difference.
    """

    gray = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    noise = cv2.absdiff(
        gray,
        blurred
    )

    noise_mean = float(
        np.mean(noise)
    )

    noise_std = float(
        np.std(noise)
    )

    return {
        "noiseImage": noise,
        "noiseMean": round(noise_mean, 2),
        "noiseStd": round(noise_std, 2)
    }
    
    
def analyze_blur(image_cv):
    """
    Analyze image sharpness using
    Variance of Laplacian.
    """

    gray = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    blur_score = float(
        laplacian.var()
    )

    return {
        "blurScore": round(
            blur_score,
            2
        )
    }
    
    
def analyze_sharpness(image_cv):
    """
    Analyze image sharpness using Sobel Gradient.
    """

    gray = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2GRAY
    )

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(
        sobel_x ** 2 + sobel_y ** 2
    )

    sharpness_score = float(
        np.mean(gradient)
    )

    return {
        "sharpnessScore": round(
            sharpness_score,
            2
        )
    }
    
    
def analyze_local_blocks(image_cv):
    """
    Analyze image block-by-block
    to detect suspicious regions.
    """

    gray = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2GRAY
    )

    block_size = 64

    block_scores = []

    height, width = gray.shape

    for y in range(0, height, block_size):

        for x in range(0, width, block_size):

            block = gray[
                y:y + block_size,
                x:x + block_size
            ]

            if block.size == 0:
                continue

            score = cv2.Laplacian(
                block,
                cv2.CV_64F
            ).var()

            block_scores.append(score)

    average_score = float(
        np.mean(block_scores)
    )

    std_score = float(
        np.std(block_scores)
    )

    return {
        "blockSharpnessAverage": round(
            average_score,
            2
        ),
        "blockSharpnessStd": round(
            std_score,
            2
        )
    }
    
    
def analyze_forgery(image_cv):
    """
    Run all forgery feature extraction methods
    and combine the results.
    """

    edge_result = detect_edges(image_cv)

    noise_result = analyze_noise(image_cv)

    blur_result = analyze_blur(image_cv)

    sharpness_result = analyze_sharpness(image_cv)

    block_result = analyze_local_blocks(image_cv)

    return {
        "edgeDensity": edge_result["edgeDensity"],

        "noiseMean": noise_result["noiseMean"],
        "noiseStd": noise_result["noiseStd"],

        "blurScore": blur_result["blurScore"],

        "sharpnessScore": sharpness_result["sharpnessScore"],

        "blockSharpnessAverage": block_result["blockSharpnessAverage"],
        "blockSharpnessStd": block_result["blockSharpnessStd"]
    }