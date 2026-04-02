from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CSV = Path("outputs/pipeline/sweeps/db_heuristic_grid_search/best_heuristic_scored_players.csv")
DEFAULT_OUTPUT_CSV = Path("artifacts/db_model_preprocessed.csv")
DEFAULT_MANIFEST_PATH = Path("artifacts/feature_manifest.json")
RANDOM_SEED = 42
TARGET_COLUMN = "NFL_production_value"

DB_POSITIONS = {
    "CB",
    "DB",
    "FS",
    "NB",
    "S",
    "SAF",
    "SAFETY",
    "SS",
    "CORNER",
    "CORNERBACK",
}

BASE_COMBINE_FEATURE_MAP = {
    "40yd": "forty_yard_sec",
    "Vertical": "vertical_jump_in",
    "Bench": "bench_reps",
    "Broad Jump": "broad_jump_in",
    "3Cone": "three_cone_sec",
    "Shuttle": "shuttle_sec",
    "Ht": "height_in",
    "Wt": "weight_lb",
}

BASE_COLLEGE_FEATURE_MAP = {
    "college_games": "college_games",
    "college_interceptions": "college_interceptions",
    "college_interception_yards": "college_interception_yards",
    "college_interception_tds": "college_interception_tds",
    "college_passes_defended": "college_passes_defended",
    "college_combined_tackles": "college_combined_tackles",
    "college_solo_tackles": "college_solo_tackles",
    "college_assisted_tackles": "college_assisted_tackles",
    "college_sacks": "college_sacks",
    "college_tfl": "college_tfl",
    "college_forced_fumbles": "college_forced_fumbles",
    "college_fumble_recoveries": "college_fumble_recoveries",
    "college_fumble_recovery_yards": "college_fumble_recovery_yards",
    "college_fumble_recovery_tds": "college_fumble_recovery_tds",
}


@dataclass(frozen=True)
class SplitConfig:
    seed: int = RANDOM_SEED
    val_size: float = 0.15
    test_size: float = 0.15
    mode: str = "random"  # random | draft_year
    val_start_year: int | None = None
    test_start_year: int | None = None


def _coerce_height_to_inches(series: pd.Series) -> pd.Series:
    parsed = series.astype(str).str.extract(r"(?P<feet>\d+)-(?P<inches>\d+)")
    parsed_inches = pd.to_numeric(parsed["feet"], errors="coerce") * 12 + pd.to_numeric(
        parsed["inches"], errors="coerce"
    )
    numeric_fallback = pd.to_numeric(series, errors="coerce")
    return parsed_inches.fillna(numeric_fallback)


def _coerce_broad_jump_to_inches(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    parsed = text.str.extract(r"(?P<feet>\d+)[-\s]+(?P<inches>\d+)")
    parsed_inches = pd.to_numeric(parsed["feet"], errors="coerce") * 12 + pd.to_numeric(
        parsed["inches"], errors="coerce"
    )
    numeric_fallback = pd.to_numeric(series, errors="coerce")
    return parsed_inches.fillna(numeric_fallback)


def _is_db_position(value: object) -> bool:
    if pd.isna(value):
        return False
    normalized = str(value).upper().replace("-", "/").replace(",", "/").strip()
    tokens = {token.strip() for token in normalized.split("/") if token.strip()}
    return any(token in DB_POSITIONS for token in tokens)


def _filter_db_rows(df: pd.DataFrame) -> pd.DataFrame:
    pos_mask = df.get("Pos", pd.Series(index=df.index, dtype=object)).apply(_is_db_position)
    college_pos_mask = df.get("college_pos", pd.Series(index=df.index, dtype=object)).apply(
        _is_db_position
    )
    return df.loc[pos_mask | college_pos_mask].copy()


def _collapse_to_player_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse season-level rows to one row per draft prospect.

    Input scored-player files often contain one row per player-season.
    For the DB model we need one row per player, so we aggregate target label
    across seasons while preserving combine/college covariates.
    """
    if df.empty:
        return df.copy()

    group_keys = [column for column in ["NFL_id", "Player", "combine_year"] if column in df.columns]
    if not group_keys:
        return df.copy()

    working = df.copy()
    if TARGET_COLUMN in working.columns:
        working[TARGET_COLUMN] = pd.to_numeric(working[TARGET_COLUMN], errors="coerce")

    aggregated = (
        working.groupby(group_keys, dropna=False, as_index=False)
        .agg(
            {
                column: ("sum" if column == TARGET_COLUMN else "first")
                for column in working.columns
                if column not in group_keys
            }
        )
        .copy()
    )
    return aggregated


def _select_and_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    if "Ht" in working.columns:
        working["Ht"] = _coerce_height_to_inches(working["Ht"])
    if "Broad Jump" in working.columns:
        working["Broad Jump"] = _coerce_broad_jump_to_inches(working["Broad Jump"])

    source_features = {**BASE_COMBINE_FEATURE_MAP, **BASE_COLLEGE_FEATURE_MAP}
    for source_col in source_features:
        if source_col not in working.columns:
            working[source_col] = np.nan
        working[source_col] = pd.to_numeric(working[source_col], errors="coerce")

    selected_columns = [column for column in ["Player", "NFL_id", "Pos"] if column in working.columns] + list(
        source_features.keys()
    )
    if "combine_year" in working.columns:
        selected_columns.append("combine_year")
    if TARGET_COLUMN in working.columns:
        selected_columns.append(TARGET_COLUMN)

    out = working[selected_columns].copy()
    out = out.rename(columns=source_features)
    return out


def _add_deterministic_imputations(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float | None]]]:
    out = df.copy()
    imputation_meta: dict[str, dict[str, float | None]] = {}

    for feature in feature_columns:
        indicator = f"{feature}_missing"
        out[indicator] = out[feature].isna().astype(int)

        median_value = out[feature].median()
        fill_value = float(median_value) if pd.notna(median_value) else 0.0
        out[feature] = out[feature].fillna(fill_value)

        mean_value = out[feature].mean()
        std_value = out[feature].std(ddof=0)
        if pd.isna(std_value) or std_value == 0:
            out[f"{feature}_z"] = 0.0
            std_for_meta = None
        else:
            out[f"{feature}_z"] = (out[feature] - mean_value) / std_value
            std_for_meta = float(std_value)

        imputation_meta[feature] = {
            "median": fill_value,
            "mean": float(mean_value) if pd.notna(mean_value) else None,
            "std": std_for_meta,
        }

    return out, imputation_meta


def _split_random(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    out = df.copy()
    if not 0 < config.val_size < 1 or not 0 < config.test_size < 1:
        raise ValueError("val_size and test_size must each be between 0 and 1.")
    if config.val_size + config.test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")

    rng = np.random.default_rng(config.seed)
    shuffled = rng.permutation(out.index.to_numpy())
    n = len(shuffled)
    n_test = int(round(n * config.test_size))
    n_val = int(round(n * config.val_size))

    split = pd.Series("train", index=out.index)
    test_idx = shuffled[:n_test]
    val_idx = shuffled[n_test : n_test + n_val]

    split.loc[test_idx] = "test"
    split.loc[val_idx] = "val"
    out["dataset_split"] = split
    return out


def _split_draft_year(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    out = df.copy()
    combine_year = pd.to_numeric(out["combine_year"], errors="coerce")
    valid_years = sorted(combine_year.dropna().astype(int).unique().tolist())
    if len(valid_years) < 3:
        raise ValueError("Draft-year split requires at least three distinct combine_year values.")

    inferred_test_start = config.test_start_year
    if inferred_test_start is None:
        inferred_test_start = valid_years[max(1, int(len(valid_years) * 0.8))]

    inferred_val_start = config.val_start_year
    if inferred_val_start is None:
        inferred_val_start = valid_years[max(0, valid_years.index(inferred_test_start) - 2)]

    split = pd.Series("train", index=out.index)
    split.loc[combine_year >= inferred_val_start] = "val"
    split.loc[combine_year >= inferred_test_start] = "test"
    split.loc[combine_year.isna()] = "train"

    out["dataset_split"] = split
    out["split_val_start_year"] = inferred_val_start
    out["split_test_start_year"] = inferred_test_start
    return out


def apply_split(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    if config.mode == "random":
        return _split_random(df, config)
    if config.mode == "draft_year":
        return _split_draft_year(df, config)
    raise ValueError("Split mode must be either 'random' or 'draft_year'.")


def preprocess_db_model_dataset(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    split_config: SplitConfig = SplitConfig(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(input_csv, low_memory=False)
    db_only = _filter_db_rows(raw)
    player_level = _collapse_to_player_level(db_only)
    selected = _select_and_standardize_columns(player_level)

    base_features = list(BASE_COMBINE_FEATURE_MAP.values()) + list(
        BASE_COLLEGE_FEATURE_MAP.values()
    )
    processed, imputation_meta = _add_deterministic_imputations(selected, base_features)
    processed = apply_split(processed, split_config)

    z_features = [f"{feature}_z" for feature in base_features]
    missing_flags = [f"{feature}_missing" for feature in base_features]
    model_inputs = z_features + missing_flags

    output_columns = [
        column
        for column in ["Player", "NFL_id", "Pos", TARGET_COLUMN, "dataset_split"]
        if column in processed.columns
    ] + model_inputs
    processed = processed[output_columns].copy()

    manifest = {
        "input_csv": str(input_csv),
        "row_count": int(len(processed)),
        "split_mode": split_config.mode,
        "seed": split_config.seed,
        "feature_groups": {
            "combine": list(BASE_COMBINE_FEATURE_MAP.values()),
            "college": list(BASE_COLLEGE_FEATURE_MAP.values()),
            "missing_indicators": missing_flags,
            "standardized_features": z_features,
        },
        "model_input_features": model_inputs,
        "retained_raw_feature_columns": [],
        "imputation_and_scaling": imputation_meta,
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_csv, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return processed, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess DB scouting data for modeling.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        choices=["random", "draft_year"],
        default="random",
        help="Split strategy: random or draft_year.",
    )
    parser.add_argument("--val-start-year", type=int, default=None)
    parser.add_argument("--test-start-year", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_db_model_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        manifest_path=args.manifest,
        split_config=SplitConfig(
            seed=args.seed,
            val_size=args.val_size,
            test_size=args.test_size,
            mode=args.split_mode,
            val_start_year=args.val_start_year,
            test_start_year=args.test_start_year,
        ),
    )


if __name__ == "__main__":
    main()
