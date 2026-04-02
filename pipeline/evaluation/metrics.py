from __future__ import annotations

import numpy as np
import pandas as pd


def compute_basic_metrics(
    df: pd.DataFrame,
    score_column: str,
    target_column: str | None,
) -> dict[str, float | int | None]:
    """Compute basic scalar metrics for a scored dataset."""
    out: dict[str, float | int | None] = {
        "n_rows": int(len(df)),
        "score_mean": float(df[score_column].mean()),
        "score_std": float(df[score_column].std(ddof=0)),
    }

    if target_column and target_column in df.columns:
        subset = df[[score_column, target_column]].dropna()
        out["n_eval_rows"] = int(len(subset))
        if len(subset) >= 2:
            corr = subset[score_column].corr(subset[target_column], method="pearson")
            out["pearson_corr"] = float(corr) if corr is not None else None
            out["mae"] = float(
                np.abs(subset[score_column] - subset[target_column]).mean()
            )
        else:
            out["pearson_corr"] = None
            out["mae"] = None
    else:
        out["n_eval_rows"] = 0
        out["pearson_corr"] = None
        out["mae"] = None

    return out
