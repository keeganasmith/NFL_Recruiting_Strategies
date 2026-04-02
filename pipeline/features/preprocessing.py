from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PreparedData:
    """Container for pipeline-ready data artifacts."""

    df: pd.DataFrame
    feature_columns: list[str]


def prepare_features(df: pd.DataFrame) -> PreparedData:
    """Apply common preprocessing shared by all heuristics.

    - Normalizes positional labels.
    - Coerces combine metric columns to numeric when available.
    """
    working = df.copy()

    if "Pos" in working.columns:
        working["Pos"] = working["Pos"].astype(str).str.upper().str.strip()

    numeric_candidates = [
        "Ht",
        "Wt",
        "40yd",
        "Vertical",
        "Bench",
        "Broad Jump",
        "3Cone",
        "Shuttle",
    ]
    feature_columns: list[str] = []
    for col in numeric_candidates:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
            feature_columns.append(col)

    return PreparedData(df=working, feature_columns=feature_columns)
