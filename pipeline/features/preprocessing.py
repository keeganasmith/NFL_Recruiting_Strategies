from __future__ import annotations

import pandas as pd

NUMERIC_SOURCE_COLUMNS = [
    "defensive_gamesPlayed",
    "defensive_totalTackles",
    "defensive_sacks",
    "defensive_interceptions",
    "defensive_passesDefended",
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare defensive-back feature columns used by the NFL production heuristic.

    The pipeline intentionally uses direct all_data.csv columns rather than heuristic
    aggregates so that configs and scored outputs are schema-aligned and auditable.
    """
    working = df.copy()

    for column in NUMERIC_SOURCE_COLUMNS:
        if column not in working.columns:
            working[column] = 0.0
        working[column] = _numeric(working[column])

    return working
