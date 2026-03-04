"""Utilities for assigning players to their primary NFL position."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_position_values(series: pd.Series) -> pd.Series:
    """Trim and normalize empty/null-like position values."""
    values = series.astype(str).str.strip()
    return values.replace({"": np.nan, "nan": np.nan, "None": np.nan})


def derive_primary_nfl_positions(
    df: pd.DataFrame,
    *,
    player_key_col: str = "player_key",
    nfl_position_col: str = "position",
    fallback_position_col: str = "Pos",
    games_played_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return one primary NFL position per player using games-played weighting.

    The selected position is the one with the highest sum of games played.
    Ties are broken by the number of rows at that position, then alphabetically.
    """
    working = df.copy()

    nfl_pos = clean_position_values(
        working[nfl_position_col] if nfl_position_col in working.columns else pd.Series(index=working.index, dtype=object)
    )
    fallback_pos = clean_position_values(
        working[fallback_position_col] if fallback_position_col in working.columns else pd.Series(index=working.index, dtype=object)
    )
    working["nfl_position"] = nfl_pos.fillna(fallback_pos)

    if games_played_cols is None:
        games_played_cols = [c for c in working.columns if c.endswith("_gamesPlayed")]

    if games_played_cols:
        games = working[games_played_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    else:
        games = pd.Series(0.0, index=working.index)
    working["row_games_played"] = games

    position_summary = (
        working.dropna(subset=["nfl_position"])
        .groupby([player_key_col, "nfl_position"], dropna=False)
        .agg(position_games=("row_games_played", "sum"), position_rows=("nfl_position", "size"))
        .reset_index()
        .sort_values(
            [player_key_col, "position_games", "position_rows", "nfl_position"],
            ascending=[True, False, False, True],
        )
    )

    return position_summary.drop_duplicates(player_key_col)[[player_key_col, "nfl_position"]].rename(
        columns={"nfl_position": "primary_nfl_position"}
    )


def filter_rows_to_primary_nfl_position(
    df: pd.DataFrame,
    *,
    player_key_col: str = "player_key",
    nfl_position_col: str = "position",
    fallback_position_col: str = "Pos",
    games_played_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Keep only rows that match each player's primary NFL position."""
    working = df.copy()
    primary_positions = derive_primary_nfl_positions(
        working,
        player_key_col=player_key_col,
        nfl_position_col=nfl_position_col,
        fallback_position_col=fallback_position_col,
        games_played_cols=games_played_cols,
    )

    nfl_pos = clean_position_values(
        working[nfl_position_col] if nfl_position_col in working.columns else pd.Series(index=working.index, dtype=object)
    )
    fallback_pos = clean_position_values(
        working[fallback_position_col] if fallback_position_col in working.columns else pd.Series(index=working.index, dtype=object)
    )
    working["nfl_position"] = nfl_pos.fillna(fallback_pos)

    working = working.merge(primary_positions, on=player_key_col, how="left")
    working["primary_nfl_position"] = working["primary_nfl_position"].fillna(working["nfl_position"])
    filtered = working[working["nfl_position"] == working["primary_nfl_position"]].copy()
    filtered[fallback_position_col] = filtered["primary_nfl_position"]
    return filtered
