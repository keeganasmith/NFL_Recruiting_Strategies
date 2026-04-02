"""Plot NFL production value against draft round.

The script reads `best_heuristic_scored_players.csv`, derives a numeric
`draft_round` from the `Drafted (tm/rnd/yr)` column, and creates a scatter plot
of `NFL_production_value` vs `draft_round`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROUND_PATTERN = re.compile(r"/\s*(\d+)(?:st|nd|rd|th)\s*/", re.IGNORECASE)


def extract_round(drafted_value: str) -> float:
    """Extract the numeric draft round from a drafted string."""
    if pd.isna(drafted_value):
        return float("nan")

    match = ROUND_PATTERN.search(str(drafted_value))
    if not match:
        return float("nan")

    return float(match.group(1))


def build_plot(input_csv: Path, output_png: Path) -> None:
    """Read source CSV, derive draft rounds, and save scatter plot."""
    df = pd.read_csv(input_csv)

    required_columns = {"Drafted (tm/rnd/yr)", "NFL_production_value"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {missing_cols}")

    df["draft_round"] = df["Drafted (tm/rnd/yr)"].apply(extract_round)
    plot_df = df.dropna(subset=["draft_round", "NFL_production_value"]).copy()

    if plot_df.empty:
        raise ValueError("No rows available to plot after deriving draft rounds.")

    plt.figure(figsize=(10, 6))
    plt.scatter(
        plot_df["draft_round"],
        plot_df["NFL_production_value"],
        alpha=0.35,
        edgecolors="none",
    )
    plt.title("NFL Production Value vs Draft Round")
    plt.xlabel("Draft Round")
    plt.ylabel("NFL Production Value")
    plt.xticks(sorted(plot_df["draft_round"].astype(int).unique()))
    plt.grid(alpha=0.2)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    default_input = Path("outputs/pipeline/sweeps/db_heuristic_grid_search/best_heuristic_scored_players.csv")
    default_output = Path("outputs/visualizations/nfl_production_vs_draft_round.png")

    parser = argparse.ArgumentParser(
        description="Plot NFL production value versus derived draft round.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_input,
        help=f"Path to source CSV (default: {default_input})",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=default_output,
        help=f"Path for output plot image (default: {default_output})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_plot(args.input_csv, args.output_png)
    print(f"Saved plot to: {args.output_png}")


if __name__ == "__main__":
    main()
