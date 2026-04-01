#!/usr/bin/env python3
"""Build a unified player dataset from pre-2000 defensive and combine stats CSV files.

The script performs an inner join so only players present in BOTH input files are
included in the output, which satisfies the requirement to exclude players found
only in `combine_with_stats.csv`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine defensive pre-2000 player data with combine stats and write "
            "a unified all_data.csv file."
        )
    )
    parser.add_argument(
        "--defensive-csv",
        default="extension_and_data/sportsref_final_season_data.csv",
        help="Path to defensive players CSV.",
    )
    parser.add_argument(
        "--combine-csv",
        default="NFL_data/combine_with_stats.csv",
        help="Path to combine + stats CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="all_data.csv",
        help="Path to write the unified CSV.",
    )
    parser.add_argument(
        "--join-key",
        default="player",
        help="Column name used to match players across both files (case-insensitive lookup, default: player).",
    )
    return parser.parse_args()


def _normalized_player_key(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )



def _resolve_column_name(df: pd.DataFrame, requested: str, dataset_label: str) -> str:
    """Resolve a column name case-insensitively and preserve the on-disk casing."""
    by_lower = {col.lower(): col for col in df.columns}
    key = requested.lower()
    if key not in by_lower:
        raise KeyError(
            f"Join key '{requested}' not found in {dataset_label} CSV columns: "
            f"{list(df.columns)}"
        )
    return by_lower[key]

def build_unified_dataset(
    defensive_csv: Path,
    combine_csv: Path,
    output_csv: Path,
    join_key: str,
) -> pd.DataFrame:
    if not defensive_csv.exists():
        raise FileNotFoundError(f"Defensive CSV not found: {defensive_csv}")
    if not combine_csv.exists():
        raise FileNotFoundError(f"Combine CSV not found: {combine_csv}")

    defensive_df = pd.read_csv(defensive_csv, low_memory=False)
    combine_df = pd.read_csv(combine_csv, low_memory=False)

    defensive_join_col = _resolve_column_name(defensive_df, join_key, "defensive")
    combine_join_col = _resolve_column_name(combine_df, join_key, "combine")

    defensive_df = defensive_df.copy()
    combine_df = combine_df.copy()
    defensive_df["__join_key"] = _normalized_player_key(defensive_df[defensive_join_col])
    combine_df["__join_key"] = _normalized_player_key(combine_df[combine_join_col])

    defensive_df = defensive_df.loc[defensive_df["__join_key"] != ""].drop_duplicates(
        subset="__join_key"
    )
    combine_df = combine_df.loc[combine_df["__join_key"] != ""]

    merged_df = combine_df.merge(
        defensive_df,
        how="inner",
        on="__join_key",
        suffixes=("", "_defensive"),
    ).drop(columns=["__join_key"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_csv, index=False)

    logging.info("Defensive rows: %s", len(defensive_df))
    logging.info("Combine rows: %s", len(combine_df))
    logging.info("Unified rows written: %s", len(merged_df))
    logging.info("Output: %s", output_csv)

    return merged_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    build_unified_dataset(
        defensive_csv=Path(args.defensive_csv),
        combine_csv=Path(args.combine_csv),
        output_csv=Path(args.output_csv),
        join_key=args.join_key,
    )


if __name__ == "__main__":
    main()
