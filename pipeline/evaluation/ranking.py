from __future__ import annotations

import numpy as np
import pandas as pd


def top_players(
    df: pd.DataFrame,
    score_col: str = "NFL_production_value",
    top_n: int = 25,
) -> pd.DataFrame:
    cols = [c for c in ["Player", "Pos", "combine_year", score_col] if c in df.columns]
    return df.sort_values(score_col, ascending=False)[cols].head(top_n).reset_index(drop=True)


def rank_stability(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str = "Player",
    score_col: str = "NFL_production_value",
    rank_top_n: int = 100,
) -> pd.DataFrame:
    train_rank = (
        train_df[[id_col, score_col]]
        .groupby(id_col, as_index=False)[score_col]
        .mean()
        .sort_values(score_col, ascending=False)
        .head(rank_top_n)
    )
    test_rank = (
        test_df[[id_col, score_col]]
        .groupby(id_col, as_index=False)[score_col]
        .mean()
        .sort_values(score_col, ascending=False)
        .head(rank_top_n)
    )

    merged = train_rank.merge(test_rank, on=id_col, how="inner", suffixes=("_train", "_test"))
    if merged.empty:
        corr = np.nan
        overlap = 0.0
    else:
        train_rank_values = merged["NFL_production_value_train"].rank(method="average")
        test_rank_values = merged["NFL_production_value_test"].rank(method="average")
        corr = train_rank_values.corr(test_rank_values, method="pearson")
        overlap = len(merged) / float(rank_top_n)

    return pd.DataFrame(
        [
            {
                "train_size": int(len(train_df)),
                "test_size": int(len(test_df)),
                "top_n": int(rank_top_n),
                "top_n_overlap_rate": float(overlap),
                "spearman_rank_corr": float(corr) if pd.notna(corr) else np.nan,
            }
        ]
    )
