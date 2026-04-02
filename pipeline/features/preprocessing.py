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


def restrict_to_early_career(df: pd.DataFrame, max_career_year: int = 4) -> pd.DataFrame:
    """Keep only rows from the first `max_career_year` NFL seasons for each player.

    Career year is derived as `(season_year - combine_year + 1)`. Rows without
    both year fields are left untouched to avoid silently dropping incomplete data.
    """
    working = df.copy()
    if "season_year" not in working.columns or "combine_year" not in working.columns:
        return working

    season_year = pd.to_numeric(working["season_year"], errors="coerce")
    combine_year = pd.to_numeric(working["combine_year"], errors="coerce")
    career_year = season_year - combine_year + 1

    mask_unknown = season_year.isna() | combine_year.isna()
    mask_first_contract = career_year <= float(max_career_year)
    keep_mask = mask_unknown | mask_first_contract

    out = working.loc[keep_mask].copy()
    out["career_year"] = career_year.loc[keep_mask]
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare defensive-back feature columns used by the NFL production heuristic.

    The pipeline intentionally uses direct all_data.csv columns rather than heuristic
    aggregates so that configs and scored outputs are schema-aligned and auditable.
    """
    working = restrict_to_early_career(df, max_career_year=4)

    for column in NUMERIC_SOURCE_COLUMNS:
        if column not in working.columns:
            working[column] = 0.0
        working[column] = _numeric(working[column])

    return working
