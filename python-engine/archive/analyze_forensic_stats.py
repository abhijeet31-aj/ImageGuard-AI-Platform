import csv
import statistics
from pathlib import Path


INPUT_FILE = Path("forensic_100_results.csv")


FEATURES = [
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


def load_data():

    with open(
        INPUT_FILE,
        encoding="utf-8"
    ) as f:

        return list(csv.DictReader(f))


def stats(rows, feature):

    values = [
        float(row[feature])
        for row in rows
    ]

    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values)
            if len(values) > 1 else 0
    }


def main():

    rows = load_data()

    ai_rows = [
        r for r in rows
        if r["true_label"] == "AI"
    ]

    real_rows = [
        r for r in rows
        if r["true_label"] == "REAL"
    ]

    print()
    print("=" * 120)
    print("IMAGEGUARD - FORENSIC STATISTICAL ANALYSIS")
    print("=" * 120)

    print()
    print(f"Total images : {len(rows)}")
    print(f"AI images    : {len(ai_rows)}")
    print(f"REAL images  : {len(real_rows)}")

    print()
    print("-" * 120)

    print(
        f"{'FEATURE':<28}"
        f"{'AI MEAN':>12}"
        f"{'REAL MEAN':>12}"
        f"{'AI MEDIAN':>12}"
        f"{'REAL MEDIAN':>14}"
        f"{'SEPARATION':>14}"
    )

    print("-" * 120)

    for feature in FEATURES:

        ai = stats(
            ai_rows,
            feature
        )

        real = stats(
            real_rows,
            feature
        )

        ai_mean = ai["mean"]
        real_mean = real["mean"]

        if max(ai_mean, real_mean) != 0:
            separation = (
                abs(ai_mean - real_mean)
                / max(ai_mean, real_mean)
            ) * 100
        else:
            separation = 0

        print(
            f"{feature:<28}"
            f"{ai_mean:>12.2f}"
            f"{real_mean:>12.2f}"
            f"{ai['median']:>12.2f}"
            f"{real['median']:>14.2f}"
            f"{separation:>13.2f}%"
        )

    print("-" * 120)

    print()
    print("=" * 120)
    print("DETAILED RANGE")
    print("=" * 120)

    for feature in FEATURES:

        ai = stats(
            ai_rows,
            feature
        )

        real = stats(
            real_rows,
            feature
        )

        print()
        print(feature)

        print(
            f"  AI   : "
            f"min={ai['min']:.2f}, "
            f"max={ai['max']:.2f}, "
            f"mean={ai['mean']:.2f}, "
            f"median={ai['median']:.2f}, "
            f"std={ai['std']:.2f}"
        )

        print(
            f"  REAL : "
            f"min={real['min']:.2f}, "
            f"max={real['max']:.2f}, "
            f"mean={real['mean']:.2f}, "
            f"median={real['median']:.2f}, "
            f"std={real['std']:.2f}"
        )

    print()
    print("=" * 120)
    print("ANALYSIS COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()