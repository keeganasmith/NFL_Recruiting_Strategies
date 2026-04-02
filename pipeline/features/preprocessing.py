from __future__ import annotations

import pandas as pd

NUMERIC_SOURCE_COLUMNS = [
    "passing_passingYards",
    "rushing_rushingYards",
    "receiving_receivingYards",
    "passing_passingTouchdowns",
    "rushing_rushingTouchdowns",
    "receiving_receivingTouchdowns",
    "scoring_totalTouchdowns",
    "defensive_totalTackles",
    "defensive_sacks",
    "defensive_interceptions",
    "defensive_passesDefended",
    "returning_kickReturnYards",
    "returning_puntReturnYards",
    "kicking_totalKickingPoints",
    "punting_puntYards",
    "passing_gamesPlayed",
    "rushing_gamesPlayed",
    "receiving_gamesPlayed",
    "defensive_gamesPlayed",
    "scoring_gamesPlayed",
    "returning_gamesPlayed",
    "kicking_gamesPlayed",
    "punting_gamesPlayed",
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare shared derived features used by all NFL production heuristics."""
    working = df.copy()

    for column in NUMERIC_SOURCE_COLUMNS:
        if column not in working.columns:
            working[column] = 0.0
        working[column] = _numeric(working[column])

    working["offense_yards"] = (
        working["passing_passingYards"]
        + working["rushing_rushingYards"]
        + working["receiving_receivingYards"]
    )
    working["touchdowns"] = (
        working["scoring_totalTouchdowns"]
        + working["passing_passingTouchdowns"]
        + working["rushing_rushingTouchdowns"]
        + working["receiving_receivingTouchdowns"]
    )
    working["defense_impact"] = (
        working["defensive_totalTackles"]
        + (2.0 * working["defensive_sacks"])
        + (3.0 * working["defensive_interceptions"])
        + (1.5 * working["defensive_passesDefended"])
    )
    working["special_teams_impact"] = (
        working["returning_kickReturnYards"]
        + working["returning_puntReturnYards"]
        + working["kicking_totalKickingPoints"]
        + (0.05 * working["punting_puntYards"])
    )
    working["games_played_any"] = working[
        [
            "passing_gamesPlayed",
            "rushing_gamesPlayed",
            "receiving_gamesPlayed",
            "defensive_gamesPlayed",
            "scoring_gamesPlayed",
            "returning_gamesPlayed",
            "kicking_gamesPlayed",
            "punting_gamesPlayed",
        ]
    ].max(axis=1)
    working["availability_factor"] = (working["games_played_any"] / 17.0).clip(
        lower=0.0, upper=1.5
    )

    return working
