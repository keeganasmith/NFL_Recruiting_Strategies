#!/usr/bin/env python3
"""Report how often GBM-Huber above-median predictions are correct."""

from __future__ import annotations

import argparse
import csv
import math
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


def load_rows(input_csv: Path, split: str, *, required: bool = True) -> list[tuple[float, float]]:
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

    if not rows and required:
        raise ValueError(f"No rows found for dataset_split={split!r} in {input_csv}")

    return rows


def compute_mae_and_r2(rows: list[tuple[float, float]]) -> tuple[float, float]:
    actual_values = [actual for actual, _ in rows]
    predicted_values = [predicted for _, predicted in rows]

    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(actual_values, predicted_values)]
    mae = sum(absolute_errors) / len(absolute_errors)

    mean_actual = sum(actual_values) / len(actual_values)
    ss_res = sum((actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values))
    ss_tot = sum((actual - mean_actual) ** 2 for actual in actual_values)
    r2 = float("nan") if math.isclose(ss_tot, 0.0) else 1 - (ss_res / ss_tot)

    return mae, r2


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv, args.split)
    training_rows = load_rows(args.input_csv, "train", required=False)

    actual_values = [actual for actual, _ in rows]
    median_actual = statistics.median(actual_values)

    predicted_above = [(actual, pred) for actual, pred in rows if pred > median_actual]
    true_positives = [(actual, pred) for actual, pred in predicted_above if actual > median_actual]

    precision = len(true_positives) / len(predicted_above) if predicted_above else 0.0
    joint_rate = len(true_positives) / len(rows)
    training_metrics = compute_mae_and_r2(training_rows) if training_rows else None

    print(f"Rows evaluated: {len(rows)}")
    print(f"Median actual NFL_production_value ({args.split} split): {median_actual:.6f}")
    print(f"Predicted above median actual threshold: {len(predicted_above)}")
    print(f"Predicted above median and actually above median: {len(true_positives)}")
    print(f"Hit rate among above-median predictions (precision): {precision:.6%}")
    print(f"Share of all rows that are above-median hits: {joint_rate:.6%}")
    if training_metrics is None:
        print("GBM-Huber training MAE: unavailable (no rows with dataset_split='train').")
        print("GBM-Huber training R^2: unavailable (no rows with dataset_split='train').")
    else:
        training_mae, training_r2 = training_metrics
        print(f"GBM-Huber training MAE: {training_mae:.6f}")
        print(
            "GBM-Huber training R^2: "
            f"{training_r2:.6f}" if not math.isnan(training_r2) else "GBM-Huber training R^2: nan"
        )


if __name__ == "__main__":
    main()
