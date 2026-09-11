import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# 1. Load AI predictions + forensic results
# --------------------------------------------------

ai = pd.read_csv("new_test_predictions.csv")
forensic = pd.read_csv("forensic_100_results.csv")

df = ai[["filename", "true_label", "ai_probability"]].merge(
    forensic.drop(columns=["true_label"]),
    on="filename",
    how="inner"
)


# --------------------------------------------------
# 2. Features
# --------------------------------------------------

features = [
    "ai_probability",
    "meanDifference",
    "maxDifference",
    "standardDeviation",
    "edgeDensity",
    "noiseMean",
    "noiseStd",
    "blurScore",
    "sharpnessScore",
    "blockSharpnessAverage",
    "blockSharpnessStd"
]

X = df[features]
y = (df["true_label"].str.upper() == "AI").astype(int)


# --------------------------------------------------
# 3. 5-fold cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

probability = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba"
)[:, 1]

prediction = (probability >= 0.5).astype(int)


# --------------------------------------------------
# 4. Find errors
# --------------------------------------------------

df["fusion_probability"] = probability
df["fusion_prediction"] = prediction

df["fusion_prediction_label"] = df["fusion_prediction"].map({
    0: "REAL",
    1: "AI"
})

df["correct"] = (
    df["fusion_prediction"] == y
)

errors = df[~df["correct"]].copy()


# --------------------------------------------------
# 5. Display errors
# --------------------------------------------------

print("=" * 100)
print("IMAGEGUARD - FUSION ERROR ANALYSIS")
print("=" * 100)

print(f"Total errors: {len(errors)}")
print()

for _, r in errors.iterrows():

    print("-" * 100)

    print(f"Filename          : {r['filename']}")
    print(f"True Label        : {r['true_label']}")
    print(f"AI Probability    : {r['ai_probability'] * 100:.2f}%")
    print(f"Fusion Probability: {r['fusion_probability'] * 100:.2f}%")
    print(f"Prediction        : {r['fusion_prediction_label']}")

    print()
    print("FORENSIC FEATURES")

    print(f"ELA Mean          : {r['meanDifference']}")
    print(f"ELA Max           : {r['maxDifference']}")
    print(f"ELA Std           : {r['standardDeviation']}")

    print(f"Edge Density      : {r['edgeDensity']}")
    print(f"Noise Mean        : {r['noiseMean']}")
    print(f"Noise Std         : {r['noiseStd']}")

    print(f"Blur Score        : {r['blurScore']}")
    print(f"Sharpness Score   : {r['sharpnessScore']}")

    print(
        f"Block Sharpness   : "
        f"{r['blockSharpnessAverage']}"
    )

    print(
        f"Block Sharpness Std: "
        f"{r['blockSharpnessStd']}"
    )

print("-" * 100)
print("ERROR ANALYSIS COMPLETE")
print("=" * 100)