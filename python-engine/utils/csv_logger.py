import csv
import os


CSV_FILE = "dataset/ela_metrics.csv"


def save_ela_metrics(
    filename,
    category,
    mean_difference,
    max_difference,
    standard_deviation
):
    """
    Save ELA metrics into CSV.
    """

    os.makedirs("dataset", exist_ok=True)

    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Filename",
                "Category",
                "MeanDifference",
                "MaxDifference",
                "StandardDeviation"
            ])

        writer.writerow([
            filename,
            category,
            mean_difference,
            max_difference,
            standard_deviation
        ])