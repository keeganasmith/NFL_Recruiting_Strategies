"""Plot the distribution of NFL production values using unique players only.

The script reads a CSV containing an ``NFL_production_value`` column, deduplicates
rows so each player is counted exactly once, and creates a histogram (with
optional KDE overlay) to visualize the distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _resolve_player_id_column(df: pd.DataFrame) -> str:
    """Return the best available player identifier column from known candidates."""
    candidates = ("Player-additional", "NFL_id", "player", "Player")
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        "Unable to identify a player ID column. Expected one of: "
        "Player-additional, NFL_id, player, Player."
    )


def build_plot(input_csv: Path, output_png: Path, bins: int = 40, with_kde: bool = True) -> None:
    """Read the input CSV and save a distribution plot for NFL production values."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "NFL_production_value" not in df.columns:
        raise ValueError("Missing required column: NFL_production_value")

    player_id_column = _resolve_player_id_column(df)
    deduped = (
        df.assign(NFL_production_value=pd.to_numeric(df["NFL_production_value"], errors="coerce"))
        .dropna(subset=["NFL_production_value"])
        .groupby(player_id_column, as_index=False)["NFL_production_value"]
        .max()
    )

    unique_player_count = deduped[player_id_column].nunique()
    print(f"Unique players counted: {unique_player_count}")

    values = deduped["NFL_production_value"]
    if values.empty:
        raise ValueError("No valid NFL_production_value rows available to plot.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        values,
        bins=bins,
        color="#4e79a7",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.8,
        density=True,
        label="Histogram",
    )

    if with_kde:
        values.plot(kind="kde", ax=ax, color="#e15759", linewidth=2.0, label="KDE")

    ax.axvline(values.median(), color="#59a14f", linestyle="--", linewidth=1.8, label="Median")
    ax.set_title("Distribution of NFL Production Values")
    ax.set_xlabel("NFL Production Value")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_input = Path("outputs/pipeline/sweeps/db_heuristic_grid_search/best_heuristic_scored_players.csv")
    default_output = Path("outputs/visualizations/nfl_production_distribution.png")

    parser = argparse.ArgumentParser(description="Plot the distribution of NFL production values.")
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
        help=f"Path for output image (default: {default_output})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins (default: 40)",
    )
    parser.add_argument(
        "--no-kde",
        action="store_true",
        help="Disable KDE line overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_plot(
        input_csv=args.input_csv,
        output_png=args.output_png,
        bins=args.bins,
        with_kde=not args.no_kde,
    )
    print(f"Saved plot to: {args.output_png}")


if __name__ == "__main__":
    main()
