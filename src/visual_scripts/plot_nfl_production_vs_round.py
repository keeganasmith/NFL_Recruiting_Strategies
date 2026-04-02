"""Plot NFL production value against draft round.

The script reads `best_heuristic_scored_players.csv`, derives a numeric
`draft_round` from the `Drafted (tm/rnd/yr)` column, and creates a
distribution-focused visualization of `NFL_production_value` by round.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    """Read source CSV, derive draft rounds, and save distribution-focused plot."""
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

    rounds = sorted(plot_df["draft_round"].astype(int).unique())
    grouped_values = [
        plot_df.loc[plot_df["draft_round"] == round_value, "NFL_production_value"].to_numpy()
        for round_value in rounds
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    box = ax.boxplot(
        grouped_values,
        positions=rounds,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#8B0000", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.2},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#8FB9E6")
        patch.set_alpha(0.55)

    # Add jittered points so density/outliers remain visible without dominating.
    rng = np.random.default_rng(7)
    for round_value, values in zip(rounds, grouped_values):
        x_jitter = rng.normal(round_value, 0.055, size=len(values))
        ax.scatter(
            x_jitter,
            values,
            alpha=0.16,
            s=14,
            color="#1f77b4",
            edgecolors="none",
            zorder=2,
        )

    medians = [np.median(values) for values in grouped_values]
    ax.plot(rounds, medians, color="#8B0000", marker="o", linewidth=1.8, label="Round median", zorder=3)

    ax.set_title("NFL Production Value Distribution by Draft Round")
    ax.set_xlabel("Draft Round")
    ax.set_ylabel("NFL Production Value")
    ax.set_xticks(rounds)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


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
