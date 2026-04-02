"""Create a DB-only version of combine_with_stats.csv.

Usage:
    python src/data/filter_db_players.py \
        --input NFL_data/combine_with_stats.csv \
        --output NFL_data/combine_with_stats_db_only.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Common defensive back labels found in college/combine datasets.
DB_POSITIONS = {
    "CB",
    "DB",
    "FS",
    "SS",
    "S",
    "NB",
    "SAF",
    "SAFETY",
    "CORNER",
    "CORNERBACK",
}


def _is_db_position(position_value: object) -> bool:
    """Return True when the provided position maps to a defensive back role."""
    if pd.isna(position_value):
        return False

    raw = str(position_value).upper().strip()
    if not raw:
        return False

    # Some rows may encode multiple positions (e.g., "CB/S" or "CB, S").
    normalized = raw.replace("-", "/").replace(",", "/")
    tokens = {token.strip() for token in normalized.split("/") if token.strip()}

    return any(token in DB_POSITIONS for token in tokens)


def build_db_only_csv(input_csv: Path, output_csv: Path) -> int:
    """Read the combine+stats CSV, filter to DB positions, and write the result."""
    df = pd.read_csv(input_csv, low_memory=False)

    if "Pos" not in df.columns:
        raise ValueError("Input CSV must include a 'Pos' column.")

    db_df = df[df["Pos"].apply(_is_db_position)].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    db_df.to_csv(output_csv, index=False)
    return len(db_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a copy of combine_with_stats.csv containing only players "
            "who played defensive back (DB-family positions) in college."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("NFL_data/combine_with_stats.csv"),
        help="Path to the source CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("NFL_data/combine_with_stats_db_only.csv"),
        help="Path for the filtered output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_count = build_db_only_csv(args.input, args.output)
    print(f"Wrote {row_count} DB rows to {args.output}")


if __name__ == "__main__":
    main()
