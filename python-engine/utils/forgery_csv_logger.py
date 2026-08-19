import csv
import os


CSV_FILE = "dataset/forgery_metrics.csv"


def save_forgery_metrics(
    filename,
    category,
    edge_density,
    noise_mean,
    noise_std,
    blur_score,
    sharpness_score,
    block_average,
    block_std
):
    """
    Save forgery analysis metrics into CSV.
    """

    os.makedirs("dataset", exist_ok=True)

    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Filename",
                "Category",
                "EdgeDensity",
                "NoiseMean",
                "NoiseStd",
                "BlurScore",
                "SharpnessScore",
                "BlockSharpnessAverage",
                "BlockSharpnessStd"
            ])

        writer.writerow([
            filename,
            category,
            edge_density,
            noise_mean,
            noise_std,
            blur_score,
            sharpness_score,
            block_average,
            block_std
        ])