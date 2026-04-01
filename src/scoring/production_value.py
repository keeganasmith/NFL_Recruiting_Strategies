"""Standalone production-value scoring layer for NFL outcomes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "config" / "production_value_config.json"
)
REQUIRED_CONFIG_KEYS = {
    "version",
    "components",
    "position_overrides",
    "time_decay",
    "missing_data",
    "winsorization",
    "output",
}
SUPPORTED_MISSING_STRATEGIES = {"impute", "drop", "cap"}


def load_production_value_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load scoring config from a single project entrypoint.

    Supports JSON by default and YAML when PyYAML is available.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Production value config not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML configs.") from exc
        return yaml.safe_load(path.read_text())

    raise ValueError(f"Unsupported config format: {suffix}")


def _validate_config(config: Mapping[str, Any]) -> None:
    missing_keys = REQUIRED_CONFIG_KEYS - set(config.keys())
    if missing_keys:
        raise ValueError(f"Config missing required keys: {sorted(missing_keys)}")

    if not isinstance(config["components"], Mapping) or not config["components"]:
        raise ValueError("Config 'components' must be a non-empty mapping.")

    strategy = config["missing_data"].get("strategy")
    if strategy not in SUPPORTED_MISSING_STRATEGIES:
        raise ValueError(f"Unsupported missing_data.strategy: {strategy}")

    for metric, spec in config["components"].items():
        if "weight" not in spec:
            raise ValueError(f"Component '{metric}' must define weight.")
        if "transform" not in spec:
            raise ValueError(f"Component '{metric}' must define transform.")


def _validate_inputs(df: pd.DataFrame, config: Mapping[str, Any]) -> None:
    required_cols = set(config["components"].keys()) | {
        config["time_decay"]["career_year_column"],
        "Pos",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Input dataframe missing required columns: {missing_cols}")

    for metric, spec in config["components"].items():
        series = pd.to_numeric(df[metric], errors="coerce")
        min_allowed = spec.get("min")
        max_allowed = spec.get("max")
        if min_allowed is not None and (series < min_allowed).any(skipna=True):
            raise ValueError(
                f"Column '{metric}' has values below configured min={min_allowed}"
            )
        if max_allowed is not None and (series > max_allowed).any(skipna=True):
            raise ValueError(
                f"Column '{metric}' has values above configured max={max_allowed}"
            )


def _apply_transform(values: pd.Series, transform: str, eps: float = 1e-6) -> pd.Series:
    if transform == "identity":
        return values
    if transform == "log1p":
        return np.log1p(values.clip(lower=0))
    if transform == "sqrt":
        return np.sqrt(values.clip(lower=0))
    if transform == "square":
        return np.square(values)
    if transform == "sigmoid":
        return 1 / (1 + np.exp(-values))
    if transform == "inverse":
        return 1 / (values + eps)
    raise ValueError(f"Unsupported transform: {transform}")


def _winsorize(
    series: pd.Series, lower_q: float | None, upper_q: float | None
) -> pd.Series:
    out = series.copy()
    if lower_q is not None:
        out = out.clip(lower=out.quantile(lower_q))
    if upper_q is not None:
        out = out.clip(upper=out.quantile(upper_q))
    return out


def _time_decay_factor(
    career_year: pd.Series, time_decay_cfg: Mapping[str, Any]
) -> pd.Series:
    early_boost = float(time_decay_cfg.get("early_career_multiplier", 1.0))
    late_decay = float(time_decay_cfg.get("late_career_multiplier", 1.0))
    early_cutoff = int(time_decay_cfg.get("early_career_cutoff_year", 3))
    late_start = int(time_decay_cfg.get("late_career_start_year", 8))

    factor = pd.Series(np.ones(len(career_year)), index=career_year.index, dtype=float)
    factor = factor.where(career_year > early_cutoff, early_boost)
    factor = factor.where(career_year < late_start, late_decay)
    return factor


def _position_weight_for_metric(
    metric: str, pos: str, config: Mapping[str, Any]
) -> float:
    base_weight = float(config["components"][metric]["weight"])
    override = (
        config["position_overrides"]
        .get(pos, {})
        .get("components", {})
        .get(metric, {})
        .get("weight")
    )
    return float(override) if override is not None else base_weight


def _prepare_batch(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for metric in config["components"]:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")

    missing_cfg = config["missing_data"]
    strategy = missing_cfg["strategy"]

    if strategy == "drop":
        out = out.dropna(subset=list(config["components"].keys()))
    else:
        for metric, spec in config["components"].items():
            if strategy == "impute":
                fill_method = missing_cfg.get("impute", {}).get("method", "median")
                if fill_method == "zero":
                    fill_value = 0
                elif fill_method == "mean":
                    fill_value = out[metric].mean()
                else:
                    fill_value = out[metric].median()
                out[metric] = out[metric].fillna(fill_value)
            elif strategy == "cap":
                cap_value = spec.get("max")
                out[metric] = out[metric].fillna(
                    cap_value if cap_value is not None else 0
                )

    wins = config["winsorization"]
    if wins.get("enabled", False):
        lower_q = wins.get("lower_quantile")
        upper_q = wins.get("upper_quantile")
        for metric in config["components"]:
            out[metric] = _winsorize(out[metric], lower_q, upper_q)

    return out


def compute_production_value(
    player_row: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute production value for a single player row as a pure function."""
    row_df = pd.DataFrame([dict(player_row)])
    scored = compute_production_value_batch(row_df, config)
    if scored.empty:
        return {}
    return scored.iloc[0].to_dict()


def compute_production_value_batch(
    df: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Vectorized deterministic production-value scorer."""
    cfg = copy.deepcopy(dict(config))
    _validate_config(cfg)
    _validate_inputs(df, cfg)

    working = _prepare_batch(df, cfg)
    if working.empty:
        return working.assign(
            production_value=pd.Series(dtype=float), heuristic_version=cfg["version"]
        )

    career_col = cfg["time_decay"]["career_year_column"]
    career_year = pd.to_numeric(working[career_col], errors="coerce").fillna(0)
    time_factor = _time_decay_factor(career_year, cfg["time_decay"])

    score = pd.Series(np.zeros(len(working)), index=working.index, dtype=float)
    for metric, spec in cfg["components"].items():
        transformed = _apply_transform(working[metric], spec["transform"])
        weights = working["Pos"].apply(
            lambda pos: _position_weight_for_metric(metric, str(pos), cfg)
        )
        score = score + transformed * weights

    score = score * time_factor
    scale = float(cfg["output"].get("scale", 1.0))
    offset = float(cfg["output"].get("offset", 0.0))
    score = score * scale + offset

    result = working.copy()
    result["production_value"] = score
    result["heuristic_version"] = cfg["version"]

    sort_cols = [
        c for c in cfg["output"].get("deterministic_sort", []) if c in result.columns
    ]
    if sort_cols:
        result = result.sort_values(sort_cols, kind="mergesort")

    return result
