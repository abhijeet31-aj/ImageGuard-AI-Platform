"""
Phase 4 — shared constants + helpers for the training pipeline.

Used by extract_features.py, train_final_model.py, and
evaluate_final_model.py so all three agree on the same feature
list, class list, and data-splitting logic.

FOLDER STRUCTURE EXPECTED (put your downloaded datasets here):

    python-engine/datasets/
        authentic/              <- RAISE (or any untouched camera photos)
        manipulated/            <- CASIA v2.0 tampered images (splice + copy-move)
        partially_ai_edited/    <- CocoGlide / AutoSplice (local diffusion edits)
        ai_generated/           <- fully AI-generated images
                                   (optional — can reuse Community Forensics'
                                   own training data instead of collecting new)

Any image format cv2/PIL can open (.jpg, .jpeg, .png, .bmp, .tif) is fine.
Sub-folders inside each class folder are walked recursively, so you don't
need to flatten CASIA/RAISE's own folder layout first.

OPTIONAL groups.csv (source-image leakage protection):

    If you have real information about which images come from the SAME
    underlying source photo across classes (e.g. CASIA sometimes derives
    a tampered image from a specific authentic one), put a file at
    python-engine/datasets/groups.csv with columns:

        filename,group_id

    Any filename not listed there gets its own unique group (i.e. treated
    as an independent source) — this is a deliberately conservative
    default: without real source-image metadata, we do NOT guess which
    images might share a source, since a wrong guess would either hide
    a real leak (if we under-group) or throw away usable data (if we
    over-group). See split_dataset() below for how this is used.
"""

import csv
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "datasets"

GROUPS_FILE = DATASET_DIR / "groups.csv"

FEATURES_FILE = BASE_DIR / "phase4_dataset_features.csv"

MODEL_DIR = BASE_DIR / "models"
FINAL_MODEL_FILE = MODEL_DIR / "final_fusion_model.json"

EVALUATION_REPORT_FILE = BASE_DIR / "phase4_evaluation_report.csv"


# ============================================================
# CLASSES
# ============================================================
# Folder name (under datasets/) -> class label used everywhere else
# in the project (matches final_fusion.py's prediction strings).

CLASS_FOLDERS = {
    "authentic": "Authentic",
    "ai_generated": "AI Generated",
    "manipulated": "Manipulated",
    "partially_ai_edited": "Partially AI-Edited",
}

CLASS_NAMES = list(CLASS_FOLDERS.values())

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# PER-CLASS SAMPLING CAP
# ============================================================
# Large public datasets are rarely balanced across classes — capping
# oversized classes keeps (a) training from being biased toward
# whichever class happens to have the most images, and (b) feature
# extraction time reasonable (running the ViT model on tens of
# thousands of images can take hours on CPU).
#
# None = no cap for that class (use every image found).
# Adjust these numbers freely based on what you actually downloaded —
# there's nothing special about these specific values.

MAX_IMAGES_PER_CLASS = {
    "Authentic": 1200,
    "AI Generated": None,          # keep all — already the smallest class
    "Manipulated": 1200,
    "Partially AI-Edited": 1200,
}

SAMPLING_SEED = 42


# ============================================================
# FEATURES
# ============================================================
# Same 10 forensic features already used by the existing
# fusion_model.json, plus the two detector signals (converted to
# log-odds, same convention as train_fusion_model.py) that Phase 1/2
# introduced.

FORENSIC_FEATURES = [
    "meanDifference",
    "maxDifference",
    "standardDeviation",
    "edgeDensity",
    "noiseMean",
    "noiseStd",
    "blurScore",
    "sharpnessScore",
    "blockSharpnessAverage",
    "blockSharpnessStd",
]

FEATURE_COLUMNS = [
    "ai_log_odds",
    "manipulation_log_odds",
    *FORENSIC_FEATURES,
]

CSV_COLUMNS = [
    "filename",
    "class_label",
    "group_id",
    "ai_probability",
    "manipulationProbability",
    *FORENSIC_FEATURES,
]


# ============================================================
# LOG-ODDS HELPER (same convention as services/fusion.py)
# ============================================================

def probability_to_log_odds(probability):
    probability = np.clip(
        float(probability),
        1e-4,
        1.0 - 1e-4,
    )

    return float(np.log(
        probability / (1.0 - probability)
    ))


# ============================================================
# CSV I/O
# ============================================================

def load_feature_rows():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"{FEATURES_FILE} not found — run extract_features.py first."
        )

    with FEATURES_FILE.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_groups_lookup():
    """
    Returns {filename: group_id}. Empty if no groups.csv is provided
    — see module docstring for why that's the safe default.
    """

    if not GROUPS_FILE.exists():
        return {}

    with GROUPS_FILE.open(encoding="utf-8") as file:
        return {
            row["filename"]: row["group_id"]
            for row in csv.DictReader(file)
        }


# ============================================================
# FEATURE MATRIX BUILDING
# ============================================================

def rows_to_matrix(rows):
    """
    Convert loaded CSV rows into (X, y) — a standardization-ready
    float feature matrix and an integer class-index label array.
    """

    features = []
    labels = []

    for row in rows:

        ai_log_odds = probability_to_log_odds(
            row["ai_probability"]
        )

        manipulation_log_odds = probability_to_log_odds(
            row["manipulationProbability"]
        )

        feature_vector = [
            ai_log_odds,
            manipulation_log_odds,
            *[
                float(row[feature])
                for feature in FORENSIC_FEATURES
            ],
        ]

        features.append(feature_vector)

        labels.append(
            CLASS_TO_INDEX[row["class_label"]]
        )

    return (
        np.array(features, dtype=np.float64),
        np.array(labels, dtype=np.int64),
    )


# ============================================================
# SOURCE-IMAGE-LEVEL, STRATIFIED SPLIT
# ============================================================

def split_dataset(rows, train_frac=0.70, val_frac=0.15, seed=42):
    """
    Split rows into train/val/test.

    Group-aware: all images sharing a group_id (see groups.csv) always
    land in the SAME split, so a manipulated image and its authentic
    source (when that relationship is known) can never leak across
    train/test. Stratified per class so each split keeps roughly the
    same class balance as the full dataset.
    """

    rng = np.random.default_rng(seed)

    groups_lookup = load_groups_lookup()

    # Assign a group_id to every row (falls back to the filename
    # itself when not listed in groups.csv — see module docstring).
    for row in rows:
        row["group_id"] = groups_lookup.get(
            row["filename"],
            row["filename"],
        )

    split_assignment = {}  # group_id -> "train" | "val" | "test"

    for class_label in CLASS_NAMES:

        class_rows = [
            row for row in rows
            if row["class_label"] == class_label
        ]

        class_groups = sorted({
            row["group_id"] for row in class_rows
        })

        rng.shuffle(class_groups)

        train_cut = int(len(class_groups) * train_frac)
        val_cut = train_cut + int(len(class_groups) * val_frac)

        for group_id in class_groups[:train_cut]:
            split_assignment.setdefault(group_id, "train")

        for group_id in class_groups[train_cut:val_cut]:
            split_assignment.setdefault(group_id, "val")

        for group_id in class_groups[val_cut:]:
            split_assignment.setdefault(group_id, "test")

    for row in rows:
        row["split"] = split_assignment[row["group_id"]]

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    return train_rows, val_rows, test_rows


def print_split_summary(name, rows):
    print(f"\n{name} — {len(rows)} images")

    for class_label in CLASS_NAMES:
        count = sum(
            1 for row in rows
            if row["class_label"] == class_label
        )
        print(f"    {class_label:22s} {count}")