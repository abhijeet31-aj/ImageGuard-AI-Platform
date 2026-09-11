import json
from pathlib import Path

import numpy as np


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "fusion_model.json"


# ============================================================
# LOAD MODEL
# ============================================================

with MODEL_FILE.open("r", encoding="utf-8") as file:
    MODEL = json.load(file)


THRESHOLD = float(MODEL["threshold"])

FEATURE_NAMES = MODEL["features"]

MEAN = np.array(
    MODEL["mean"],
    dtype=np.float64,
)

STD = np.array(
    MODEL["std"],
    dtype=np.float64,
)

WEIGHTS = np.array(
    MODEL["weights"],
    dtype=np.float64,
)

BIAS = float(MODEL["bias"])


# ============================================================
# SIGMOID
# ============================================================

def sigmoid(value):
    value = np.clip(value, -500, 500)

    return 1.0 / (
        1.0 + np.exp(-value)
    )


# ============================================================
# AI PROBABILITY → LOG ODDS
# ============================================================

def probability_to_log_odds(probability):
    probability = np.clip(
        float(probability),
        1e-4,
        1.0 - 1e-4,
    )

    return np.log(
        probability /
        (1.0 - probability)
    )


# ============================================================
# FUSION PREDICTION
# ============================================================

def predict_fusion(
    ai_probability,
    forensic_analysis,
):
    """
    Generate the final fusion prediction using
    the fixed production fusion model.
    """

    # --------------------------------------------------------
    # 1. Convert AI probability to log-odds
    # --------------------------------------------------------

    ai_log_odds = probability_to_log_odds(
        ai_probability
    )


    # --------------------------------------------------------
    # 2. Build feature vector
    # --------------------------------------------------------

    feature_vector = [
        ai_log_odds,

        float(
            forensic_analysis["meanDifference"]
        ),

        float(
            forensic_analysis["maxDifference"]
        ),

        float(
            forensic_analysis["standardDeviation"]
        ),

        float(
            forensic_analysis["edgeDensity"]
        ),

        float(
            forensic_analysis["noiseMean"]
        ),

        float(
            forensic_analysis["noiseStd"]
        ),

        float(
            forensic_analysis["blurScore"]
        ),

        float(
            forensic_analysis["sharpnessScore"]
        ),

        float(
            forensic_analysis["blockSharpnessAverage"]
        ),

        float(
            forensic_analysis["blockSharpnessStd"]
        ),
    ]


    features = np.array(
        feature_vector,
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # 3. Validate feature count
    # --------------------------------------------------------

    if len(features) != len(MEAN):
        raise ValueError(
            "Fusion feature count does not match "
            "the saved production model."
        )


    # --------------------------------------------------------
    # 4. Normalize using training statistics
    # --------------------------------------------------------

    normalized_features = (
        features - MEAN
    ) / STD


    # --------------------------------------------------------
    # 5. Logistic regression
    # --------------------------------------------------------

    logit = (
        normalized_features @ WEIGHTS
        + BIAS
    )

    fusion_probability = float(
        sigmoid(logit)
    )


    # --------------------------------------------------------
    # 6. Final prediction
    # --------------------------------------------------------

    prediction = (
        "AI Generated"
        if fusion_probability >= THRESHOLD
        else "Real"
    )


    # --------------------------------------------------------
    # 7. Return result
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "ai_probability": round(
            float(ai_probability),
            4,
        ),
        "fusion_probability": round(
            fusion_probability,
            4,
        ),
        "confidence": round(
            max(
                fusion_probability,
                1.0 - fusion_probability,
            ),
            4,
        ),
        "threshold": THRESHOLD,
    }