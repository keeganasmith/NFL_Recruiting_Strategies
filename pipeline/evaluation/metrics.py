from __future__ import annotations

import pandas as pd


def score_summary(df: pd.DataFrame, score_col: str = "NFL_production_value") -> pd.DataFrame:
    series = pd.to_numeric(df[score_col], errors="coerce")
    summary = {
        "n_rows": int(len(series)),
        "n_scored": int(series.notna().sum()),
        "coverage": float(series.notna().mean()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "p50": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }
    return pd.DataFrame([summary])
