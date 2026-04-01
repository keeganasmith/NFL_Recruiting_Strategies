"""Prepare tidy five-column effect tables for visualization.

Converts modeling outputs (e.g., `outputs/modeling/feature_effects.csv`) into the
five-column tidy schema required by `effect_size_dotplot.py`:

    position_group, metric, estimate, ci_low, ci_high
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.position_models import (
    PositionModelingConfig,
    run_position_modeling_workflow,
)

REQUIRED_MODEL_COLUMNS = {
    "position_group",
    "feature",
    "estimate",
    "ci_lower",
    "ci_upper",
}


def _validate_model_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_MODEL_COLUMNS - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(
            "Input model-effects file is missing required columns: " f"{missing_cols}."
        )


def prepare_tidy_effects(
    model_effects_df: pd.DataFrame,
    include_pooled: bool = False,
    include_missing_indicators: bool = False,
    include_intercept: bool = False,
) -> pd.DataFrame:
    """Transform model feature effects into tidy five-column chart input.

    By default, this keeps only standardized combine features (`*_z`) and drops
    pooled, intercept, and missingness-indicator rows.
    """

    _validate_model_columns(model_effects_df)

    tidy = model_effects_df.copy()

    if not include_pooled:
        tidy = tidy.loc[tidy["position_group"] != "POOLED"]

    feature_mask = tidy["feature"].str.endswith("_z", na=False)
    if include_missing_indicators:
        feature_mask = feature_mask | tidy["feature"].str.endswith("_missing", na=False)
    if include_intercept:
        feature_mask = feature_mask | tidy["feature"].eq("intercept")

    tidy = tidy.loc[
        feature_mask, ["position_group", "feature", "estimate", "ci_lower", "ci_upper"]
    ].copy()
    tidy = tidy.rename(
        columns={
            "feature": "metric",
            "ci_lower": "ci_low",
            "ci_upper": "ci_high",
        }
    )

    tidy["metric"] = tidy["metric"].str.replace("_z$", "", regex=True)

    return tidy.sort_values(["position_group", "metric"]).reset_index(drop=True)


RAW_COMBINE_REQUIRED_COLUMNS = {
    "Pos",
    "Ht",
    "Wt",
    "40yd",
    "Vertical",
    "Bench",
    "Broad Jump",
    "3Cone",
    "Shuttle",
}
RAW_TARGET_REQUIRED_COLUMNS = {
    "career_year",
    "starts",
    "approximate_value",
    "snap_share",
    "seasons_active",
}


def _looks_like_combine_with_stats(df: pd.DataFrame) -> bool:
    """Heuristic check for season-level `combine_with_stats.csv` layout."""

    cols = set(df.columns)
    games_played_cols = [c for c in df.columns if c.endswith("_gamesPlayed")]
    return {"season_year", "Player", "Pos"}.issubset(cols) and len(
        games_played_cols
    ) >= 3


def _coalesce_player_id(df: pd.DataFrame) -> pd.Series:
    """Build stable player keys from NFL_id when available, fallback to player name."""

    nfl_id = df.get("NFL_id", pd.Series(index=df.index, dtype=object))
    nfl_id = nfl_id.astype(str).str.strip()
    has_id = nfl_id.notna() & nfl_id.ne("") & nfl_id.ne("N/A") & nfl_id.ne("nan")

    player_name = (
        df.get("Player", pd.Series(index=df.index, dtype=object))
        .astype(str)
        .str.strip()
    )
    return np.where(has_id, "id:" + nfl_id, "name:" + player_name)


def _derive_proxy_modeling_rows_from_combine_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate season-level NFL stats into player-level proxy production inputs.

    `combine_with_stats.csv` does not include starts/AV/snap_share, so we construct
    proxy targets from available games-played signals. This allows the modeling
    workflow to run end-to-end for exploratory effect-size charts.
    """

    working = df.copy()
    working["_player_key"] = _coalesce_player_id(working)

    # Estimate per-season participation from whichever stats family is populated.
    games_played_cols = [c for c in working.columns if c.endswith("_gamesPlayed")]
    gp_values = working[games_played_cols].apply(pd.to_numeric, errors="coerce")
    working["_season_games_played"] = gp_values.max(axis=1, skipna=True).fillna(0.0)

    working["season_year"] = pd.to_numeric(working["season_year"], errors="coerce")

    # Preserve first non-null combine values per player.
    combine_cols = [
        "combine_year",
        "Player",
        "NFL_id",
        "Pos",
        "Ht",
        "Wt",
        "40yd",
        "Vertical",
        "Bench",
        "Broad Jump",
        "3Cone",
        "Shuttle",
    ]
    base = (
        working.sort_values(["_player_key", "season_year"], na_position="last")
        .groupby("_player_key", as_index=False)
        .first()[["_player_key", *[c for c in combine_cols if c in working.columns]]]
    )

    agg = working.groupby("_player_key", as_index=False).agg(
        starts_proxy=("_season_games_played", "sum"),
        seasons_active=("season_year", "nunique"),
    )
    agg["snap_share"] = (
        (agg["starts_proxy"] / (agg["seasons_active"].replace(0, np.nan) * 17.0))
        .clip(0.0, 1.0)
        .fillna(0.0)
    )
    agg["approximate_value"] = (agg["starts_proxy"] * agg["snap_share"]).clip(lower=0.0)
    agg["production_value"] = (
        0.35 * np.log1p(agg["starts_proxy"])
        + 0.30 * np.log1p(agg["approximate_value"])
        + 0.20 * agg["snap_share"]
        + 0.15 * np.sqrt(agg["seasons_active"].clip(lower=0.0))
    )

    out = base.merge(agg, on="_player_key", how="inner")
    out = out.rename(columns={"starts_proxy": "starts"})
    return out.drop(columns=["_player_key"])


def _raw_modeling_missing_columns(df: pd.DataFrame) -> list[str]:
    """List missing columns needed to run modeling from raw player data."""

    missing = set()
    cols = set(df.columns)
    missing |= RAW_COMBINE_REQUIRED_COLUMNS - cols

    # Target can come from precomputed production_value OR scoring inputs.
    if "production_value" not in cols:
        print("got here")
        missing |= RAW_TARGET_REQUIRED_COLUMNS - cols

    return sorted(missing)


def load_or_generate_model_effects(
    input_csv: Path,
    model_output_dir: Path,
    model_version: str,
    bootstrap_iterations: int,
    min_group_size: int,
) -> pd.DataFrame:
    """Load feature effects directly, or generate them from raw player data.

    If `input_csv` already has feature-effect columns, it is returned directly.
    Otherwise, if it looks like raw player data with combine columns, the
    position modeling workflow is executed and the generated
    `feature_effects.csv` is loaded from `model_output_dir`.
    """

    source_df = pd.read_csv(input_csv, low_memory=False)
    if REQUIRED_MODEL_COLUMNS.issubset(set(source_df.columns)):
        return source_df

    missing = _raw_modeling_missing_columns(source_df)
    if missing:
        if _looks_like_combine_with_stats(source_df):
            source_df = _derive_proxy_modeling_rows_from_combine_stats(source_df)
            missing = _raw_modeling_missing_columns(source_df)
            if not missing:
                config = PositionModelingConfig(
                    model_version=model_version,
                    bootstrap_iterations=bootstrap_iterations,
                    min_group_size=min_group_size,
                )
                run_position_modeling_workflow(
                    df=source_df, output_dir=model_output_dir, config=config
                )
                feature_effects_path = model_output_dir / "feature_effects.csv"
                if not feature_effects_path.exists():
                    raise FileNotFoundError(
                        f"Modeling completed but no feature effects were found at {feature_effects_path}"
                    )
                return pd.read_csv(feature_effects_path, low_memory=False)

        raise ValueError(
            "Input is neither a model-effects file nor valid raw player data. "
            f"Missing raw-data columns: {missing}"
        )

    config = PositionModelingConfig(
        model_version=model_version,
        bootstrap_iterations=bootstrap_iterations,
        min_group_size=min_group_size,
    )
    run_position_modeling_workflow(
        df=source_df, output_dir=model_output_dir, config=config
    )
    feature_effects_path = model_output_dir / "feature_effects.csv"
    if not feature_effects_path.exists():
        raise FileNotFoundError(
            f"Modeling completed but no feature effects were found at {feature_effects_path}"
        )
    return pd.read_csv(feature_effects_path, low_memory=False)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert model feature effects CSV into tidy five-column chart input."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "Path to either model effects CSV (outputs/modeling/feature_effects.csv) "
            "or raw player dataset (e.g., NFL_data/combine_with_college_stats.csv)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/model_effects/standardized_effects.csv"),
        help="Output path for tidy five-column effects CSV.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=Path("outputs/modeling"),
        help=(
            "Directory for generated modeling outputs when input_csv is raw player data. "
            "Ignored when input_csv is already a model-effects file."
        ),
    )
    parser.add_argument(
        "--model-version",
        default="v1.0.0",
        help="Model version label to use if modeling is run from raw player data.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=200,
        help="Bootstrap iterations to use if modeling is run from raw player data.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=30,
        help="Minimum position-group sample size to use if modeling is run from raw player data.",
    )
    parser.add_argument(
        "--include-pooled",
        action="store_true",
        help="Include POOLED rows in output.",
    )
    parser.add_argument(
        "--include-missing-indicators",
        action="store_true",
        help="Include *_missing features in output.",
    )
    parser.add_argument(
        "--include-intercept",
        action="store_true",
        help="Include intercept rows in output.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    model_effects_df = load_or_generate_model_effects(
        input_csv=args.input_csv,
        model_output_dir=args.model_output_dir,
        model_version=args.model_version,
        bootstrap_iterations=args.bootstrap_iterations,
        min_group_size=args.min_group_size,
    )
    tidy = prepare_tidy_effects(
        model_effects_df=model_effects_df,
        include_pooled=args.include_pooled,
        include_missing_indicators=args.include_missing_indicators,
        include_intercept=args.include_intercept,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(args.output_csv, index=False)

    print(f"Saved tidy effects: {args.output_csv}")
    print("Columns:", ", ".join(tidy.columns.tolist()))
    print(f"Rows: {len(tidy)}")


if __name__ == "__main__":
    main()
