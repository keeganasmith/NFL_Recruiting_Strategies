from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_FEATURES = {
    "offense_yards",
    "touchdowns",
    "defense_impact",
    "special_teams_impact",
    "availability_factor",
}
SUPPORTED_ROLES = {"CB", "S", "slot", "outside"}


@dataclass(frozen=True)
class BinRule:
    min: float | None
    max: float | None
    value: float


@dataclass(frozen=True)
class HeuristicConfig:
    heuristic_id: str
    feature_weights: dict[str, float]
    thresholds: dict[str, list[BinRule]]
    role_overrides: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SplitStrategy:
    type: str
    seed: int
    time_split_year: int | None


@dataclass(frozen=True)
class SharedFilters:
    positions: list[str] | None
    era_min: int | None
    era_max: int | None
    minimum_snaps: int | None


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    input_data: str
    output_root: str
    split: SplitStrategy
    filters: SharedFilters
    calibration_bins: int
    output_naming: str
    variants: list[dict[str, Any]]



def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML configs. Install with `pip install pyyaml`."
        ) from exc

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must deserialize to a mapping/object.")
    return data



def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{field_name}' must be numeric. Got: {value!r}") from exc



def _validate_bin_rules(feature: str, bins: Any) -> list[BinRule]:
    if not isinstance(bins, list) or not bins:
        raise ValueError(f"thresholds.{feature}.bins must be a non-empty list.")

    out: list[BinRule] = []
    previous_max: float | None = None
    for idx, rule in enumerate(bins):
        if not isinstance(rule, dict):
            raise ValueError(
                f"thresholds.{feature}.bins[{idx}] must be an object with min/max/value."
            )

        min_v = rule.get("min")
        max_v = rule.get("max")
        value = _coerce_float(rule.get("value"), f"thresholds.{feature}.bins[{idx}].value")

        if min_v is not None:
            min_v = _coerce_float(min_v, f"thresholds.{feature}.bins[{idx}].min")
        if max_v is not None:
            max_v = _coerce_float(max_v, f"thresholds.{feature}.bins[{idx}].max")

        if min_v is not None and max_v is not None and min_v >= max_v:
            raise ValueError(
                f"thresholds.{feature}.bins[{idx}] has invalid bounds: min must be < max."
            )
        if previous_max is not None and min_v is not None and min_v < previous_max:
            raise ValueError(
                f"thresholds.{feature}.bins are not ordered: bins[{idx}] min={min_v} < previous max={previous_max}."
            )
        if max_v is not None:
            previous_max = max_v

        out.append(BinRule(min=min_v, max=max_v, value=value))

    return out



def load_heuristic_config(path: str | Path) -> HeuristicConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Heuristic config not found: {config_path}")

    cfg = _load_yaml(config_path)

    heuristic_id = cfg.get("heuristic_id")
    if not heuristic_id or not isinstance(heuristic_id, str):
        raise ValueError("Heuristic config must define string field 'heuristic_id'.")

    weights = cfg.get("feature_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Heuristic config must define non-empty mapping 'feature_weights'.")

    missing_weights = sorted(REQUIRED_FEATURES - set(weights.keys()))
    if missing_weights:
        raise ValueError(
            "feature_weights missing required feature columns: "
            f"{missing_weights}. Required: {sorted(REQUIRED_FEATURES)}"
        )

    normalized_weights: dict[str, float] = {}
    for feature, value in weights.items():
        weight = _coerce_float(value, f"feature_weights.{feature}")
        if not -1000.0 <= weight <= 1000.0:
            raise ValueError(
                f"feature_weights.{feature}={weight} out of allowed range [-1000, 1000]."
            )
        normalized_weights[feature] = weight

    thresholds_cfg = cfg.get("thresholds", {})
    if not isinstance(thresholds_cfg, dict):
        raise ValueError("'thresholds' must be a mapping keyed by feature name.")

    thresholds: dict[str, list[BinRule]] = {}
    for feature, threshold_spec in thresholds_cfg.items():
        if feature not in REQUIRED_FEATURES:
            raise ValueError(
                f"thresholds has unknown feature '{feature}'. Allowed: {sorted(REQUIRED_FEATURES)}"
            )
        if not isinstance(threshold_spec, dict):
            raise ValueError(f"thresholds.{feature} must be an object with a 'bins' field.")
        thresholds[feature] = _validate_bin_rules(feature, threshold_spec.get("bins"))

    role_overrides = cfg.get("role_overrides", {})
    if not isinstance(role_overrides, dict):
        raise ValueError("'role_overrides' must be a mapping if provided.")

    normalized_overrides: dict[str, dict[str, Any]] = {}
    for role, spec in role_overrides.items():
        if role not in SUPPORTED_ROLES:
            raise ValueError(
                f"Unsupported role override '{role}'. Allowed: {sorted(SUPPORTED_ROLES)}"
            )
        if not isinstance(spec, dict):
            raise ValueError(f"role_overrides.{role} must be an object.")
        role_weights = spec.get("feature_weights", {})
        if not isinstance(role_weights, dict):
            raise ValueError(f"role_overrides.{role}.feature_weights must be a mapping.")
        normalized_role_weights = {
            feature: _coerce_float(value, f"role_overrides.{role}.feature_weights.{feature}")
            for feature, value in role_weights.items()
        }
        unknown_features = sorted(set(normalized_role_weights) - REQUIRED_FEATURES)
        if unknown_features:
            raise ValueError(
                f"role_overrides.{role}.feature_weights has unknown features: {unknown_features}"
            )

        role_thresholds_cfg = spec.get("thresholds", {})
        if not isinstance(role_thresholds_cfg, dict):
            raise ValueError(f"role_overrides.{role}.thresholds must be a mapping.")
        normalized_role_thresholds: dict[str, list[BinRule]] = {}
        for feature, threshold_spec in role_thresholds_cfg.items():
            if feature not in REQUIRED_FEATURES:
                raise ValueError(
                    f"role_overrides.{role}.thresholds has unknown feature '{feature}'."
                )
            if not isinstance(threshold_spec, dict):
                raise ValueError(
                    f"role_overrides.{role}.thresholds.{feature} must be an object with 'bins'."
                )
            normalized_role_thresholds[feature] = _validate_bin_rules(
                feature, threshold_spec.get("bins")
            )

        normalized_overrides[role] = {
            "feature_weights": normalized_role_weights,
            "thresholds": normalized_role_thresholds,
        }

    return HeuristicConfig(
        heuristic_id=heuristic_id,
        feature_weights=normalized_weights,
        thresholds=thresholds,
        role_overrides=normalized_overrides,
    )



def validate_heuristic_inputs(df: pd.DataFrame, config: HeuristicConfig) -> None:
    missing_cols = sorted(REQUIRED_FEATURES - set(df.columns))
    if missing_cols:
        raise ValueError(
            "Heuristic input missing required feature columns: "
            f"{missing_cols}. Ensure prepare_features() has run."
        )

    for feature, weight in config.feature_weights.items():
        if not pd.api.types.is_numeric_dtype(df[feature]):
            coerced = pd.to_numeric(df[feature], errors="coerce")
            if coerced.isna().all():
                raise ValueError(
                    f"Column '{feature}' is non-numeric and cannot be scored with weight {weight}."
                )



def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    cfg = _load_yaml(config_path)

    experiment_id = cfg.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("Experiment config must define non-empty string field 'experiment_id'.")

    split_cfg = cfg.get("split", {})
    if not isinstance(split_cfg, dict):
        raise ValueError("'split' must be a mapping.")
    split_type = split_cfg.get("type", "random")
    if split_type not in {"random", "time"}:
        raise ValueError("split.type must be one of ['random', 'time'].")
    seed = int(split_cfg.get("seed", 42))
    time_split_year = split_cfg.get("time_split_year")
    if time_split_year is not None:
        time_split_year = int(time_split_year)

    filters_cfg = cfg.get("filters", {})
    if not isinstance(filters_cfg, dict):
        raise ValueError("'filters' must be a mapping.")
    positions = filters_cfg.get("positions")
    if positions is not None and not isinstance(positions, list):
        raise ValueError("filters.positions must be a list of position codes if provided.")

    variants = cfg.get("heuristic_variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("heuristic_variants must be a non-empty list.")
    for idx, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"heuristic_variants[{idx}] must be an object.")
        if not isinstance(variant.get("id"), str) or not variant["id"].strip():
            raise ValueError(f"heuristic_variants[{idx}] must contain non-empty string 'id'.")
        if "heuristic_config" not in variant:
            raise ValueError(
                f"heuristic_variants[{idx}] missing required field 'heuristic_config'."
            )

    return ExperimentConfig(
        experiment_id=experiment_id,
        input_data=str(cfg.get("input_data", "all_data.csv")),
        output_root=str(cfg.get("output_root", "outputs/pipeline/sweeps")),
        split=SplitStrategy(
            type=split_type,
            seed=seed,
            time_split_year=time_split_year,
        ),
        filters=SharedFilters(
            positions=positions,
            era_min=filters_cfg.get("era_min"),
            era_max=filters_cfg.get("era_max"),
            minimum_snaps=filters_cfg.get("minimum_snaps"),
        ),
        calibration_bins=int(cfg.get("calibration_bins", 10)),
        output_naming=str(cfg.get("output_naming", "{variant_id}")),
        variants=variants,
    )
