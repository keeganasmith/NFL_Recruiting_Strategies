from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HeuristicObjective:
    proxy_spearman: float
    calibration_rmse: float
    rank_corr: float
    top_overlap: float
    objective_score: float


def _safe_float(value: float | int | None, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    value_f = float(value)
    if np.isnan(value_f):
        return fallback
    return value_f


def compute_proxy_outcome(df: pd.DataFrame) -> pd.Series:
    """Independent proxy target for defensive-back NFL production quality.

    Uses percentile-scaled components with fixed (non-swept) weights so the
    objective is not directly optimized on the exact same linear form.
    """
    games = pd.to_numeric(df.get("defensive_gamesPlayed"), errors="coerce").fillna(0.0)
    tackles = pd.to_numeric(df.get("defensive_totalTackles"), errors="coerce").fillna(0.0)
    sacks = pd.to_numeric(df.get("defensive_sacks"), errors="coerce").fillna(0.0)
    interceptions = pd.to_numeric(df.get("defensive_interceptions"), errors="coerce").fillna(0.0)
    passes_defended = pd.to_numeric(df.get("defensive_passesDefended"), errors="coerce").fillna(0.0)

    # Use rates + per-feature percentile ranks to reduce scale sensitivity.
    availability = games.rank(pct=True)
    ball_hawking = (interceptions + 0.5 * passes_defended).rank(pct=True)
    disruption = (sacks + 0.1 * tackles).rank(pct=True)

    return 0.45 * availability + 0.35 * ball_hawking + 0.20 * disruption


def evaluate_heuristic(
    *,
    scored_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    score_col: str = "NFL_production_value",
) -> HeuristicObjective:
    score = pd.to_numeric(scored_df[score_col], errors="coerce").fillna(0.0)
    proxy = compute_proxy_outcome(scored_df)

    proxy_spearman = _safe_float(score.rank(method="average").corr(proxy.rank(method="average"), method="pearson"))

    bins = pd.qcut(score, q=10, labels=False, duplicates="drop")
    grouped = (
        pd.DataFrame({"bin": bins, "score": score, "proxy": proxy})
        .groupby("bin", dropna=True)
        .agg(score_mean=("score", "mean"), proxy_mean=("proxy", "mean"))
        .reset_index(drop=True)
    )
    if len(grouped) > 1:
        score_norm = grouped["score_mean"].rank(pct=True)
        proxy_norm = grouped["proxy_mean"].rank(pct=True)
        calibration_rmse = _safe_float(np.sqrt(np.mean((score_norm - proxy_norm) ** 2)))
    else:
        calibration_rmse = 1.0

    row = ranking_df.iloc[0].to_dict() if not ranking_df.empty else {}
    rank_corr = _safe_float(row.get("spearman_rank_corr"))
    top_overlap = _safe_float(row.get("top_n_overlap_rate"))

    objective_score = (
        0.55 * proxy_spearman
        + 0.25 * rank_corr
        + 0.20 * top_overlap
        - 0.15 * calibration_rmse
    )

    return HeuristicObjective(
        proxy_spearman=proxy_spearman,
        calibration_rmse=calibration_rmse,
        rank_corr=rank_corr,
        top_overlap=top_overlap,
        objective_score=float(objective_score),
    )
