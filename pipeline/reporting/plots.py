from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_score_distribution(
    df: pd.DataFrame,
    output_path: Path,
    score_col: str = "NFL_production_value",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df[score_col].dropna(), bins=40, color="#1f77b4", alpha=0.8)
    ax.set_title("NFL Production Value Distribution")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Players")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_calibration_plot(calibration_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        calibration_df["score_mean"],
        calibration_df["proxy_outcome_mean"],
        marker="o",
        linewidth=1.5,
        color="#2ca02c",
    )
    ax.set_title("Calibration: Heuristic Score vs Proxy NFL Outcome")
    ax.set_xlabel("Mean NFL_production_value (bin)")
    ax.set_ylabel("Mean proxy NFL outcome")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
