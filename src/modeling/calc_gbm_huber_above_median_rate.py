#!/usr/bin/env python3
"""Report how often GBM-Huber above-median predictions are correct."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


DEFAULT_INPUT_CSV = Path("outputs/model_training/gbm_huber_test_predictions.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate how often the GBM with Huber loss predicts that a player's "
            "NFL production value is above the median when the actual value is also above the median."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Path to GBM-Huber predictions CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Only evaluate rows where dataset_split equals this value (default: test)",
    )
    return parser.parse_args()


def load_rows(input_csv: Path, split: str) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"dataset_split", "NFL_production_value", "gbm_huber_prediction"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

        for row in reader:
            if row["dataset_split"] != split:
                continue
            actual = float(row["NFL_production_value"])
            predicted = float(row["gbm_huber_prediction"])
            rows.append((actual, predicted))

    if not rows:
        raise ValueError(f"No rows found for dataset_split={split!r} in {input_csv}")

    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv, args.split)

    actual_values = [actual for actual, _ in rows]
    median_actual = statistics.median(actual_values)

    predicted_above = [(actual, pred) for actual, pred in rows if pred > median_actual]
    true_positives = [(actual, pred) for actual, pred in predicted_above if actual > median_actual]

    precision = len(true_positives) / len(predicted_above) if predicted_above else 0.0
    joint_rate = len(true_positives) / len(rows)

    print(f"Rows evaluated: {len(rows)}")
    print(f"Median actual NFL_production_value ({args.split} split): {median_actual:.6f}")
    print(f"Predicted above median actual threshold: {len(predicted_above)}")
    print(f"Predicted above median and actually above median: {len(true_positives)}")
    print(f"Hit rate among above-median predictions (precision): {precision:.6%}")
    print(f"Share of all rows that are above-median hits: {joint_rate:.6%}")


if __name__ == "__main__":
    main()
