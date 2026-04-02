from __future__ import annotations

import pandas as pd


def add_proxy_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Create a transparent proxy NFL outcome for heuristic calibration checks."""
    working = df.copy()
    working["proxy_nfl_outcome"] = (
        0.01 * working["offense_yards"]
        + 2.0 * working["touchdowns"]
        + 0.3 * working["defense_impact"]
        + 0.005 * working["special_teams_impact"]
    )
    return working


def calibration_table(
    df: pd.DataFrame,
    score_col: str = "NFL_production_value",
    bins: int = 10,
) -> pd.DataFrame:
    working = add_proxy_outcome(df)
    quantiles = pd.qcut(working[score_col], q=bins, labels=False, duplicates="drop")
    grouped = (
        working.assign(score_bin=quantiles)
        .groupby("score_bin", dropna=True)
        .agg(
            players=(score_col, "size"),
            score_mean=(score_col, "mean"),
            proxy_outcome_mean=("proxy_nfl_outcome", "mean"),
        )
        .reset_index()
        .sort_values("score_bin")
    )
    return grouped
