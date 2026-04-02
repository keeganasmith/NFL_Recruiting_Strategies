from __future__ import annotations

import pandas as pd


def calibration_table(
    df: pd.DataFrame,
    score_column: str,
    target_column: str | None,
    bins: int = 10,
) -> pd.DataFrame:
    """Build quantile-bin calibration table when target is available."""
    if not target_column or target_column not in df.columns:
        return pd.DataFrame(
            columns=["bin", "n", "score_mean", "target_mean", "target_std"]
        )

    subset = df[[score_column, target_column]].dropna().copy()
    if subset.empty:
        return pd.DataFrame(
            columns=["bin", "n", "score_mean", "target_mean", "target_std"]
        )

    subset["bin"] = pd.qcut(
        subset[score_column], q=min(bins, subset[score_column].nunique()), duplicates="drop"
    )
    grouped = (
        subset.groupby("bin", observed=True)
        .agg(
            n=(score_column, "size"),
            score_mean=(score_column, "mean"),
            target_mean=(target_column, "mean"),
            target_std=(target_column, "std"),
        )
        .reset_index()
    )
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped
