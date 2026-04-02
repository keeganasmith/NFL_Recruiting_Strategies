from __future__ import annotations

import pandas as pd


def ranking_analysis(
    df: pd.DataFrame,
    score_column: str,
    top_k: int = 25,
    id_column: str = "Player",
) -> pd.DataFrame:
    """Return top-k ranking table for scored players."""
    ranking = df.sort_values(score_column, ascending=False).copy()
    ranking["rank"] = range(1, len(ranking) + 1)

    columns = ["rank", score_column]
    if id_column in ranking.columns:
        columns.insert(1, id_column)
    if "Pos" in ranking.columns:
        columns.append("Pos")

    return ranking.loc[:, columns].head(top_k)
