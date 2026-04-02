from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.heuristics.base import Heuristic
from pipeline.heuristics.combine_weighted import CombineWeightedHeuristic


def _build_combine_weighted(params: dict[str, Any]) -> Heuristic:
    return CombineWeightedHeuristic(
        weights=params.get("weights", {}),
        invert_columns=params.get("invert_columns", []),
    )


HEURISTIC_BUILDERS = {
    "combine_weighted": _build_combine_weighted,
}


def load_heuristic_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Heuristic config was not found: {cfg_path}")
    return json.loads(cfg_path.read_text())


def create_heuristic(config_key: str, params: dict[str, Any]) -> Heuristic:
    try:
        builder = HEURISTIC_BUILDERS[config_key]
    except KeyError as exc:
        known = ", ".join(sorted(HEURISTIC_BUILDERS))
        raise KeyError(f"Unknown heuristic '{config_key}'. Known: {known}") from exc
    return builder(params)


def build_heuristic_from_config(config: dict[str, Any]) -> Heuristic:
    heuristic_key = config.get("heuristic")
    if not heuristic_key:
        raise ValueError("Heuristic config must include 'heuristic'.")
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("Heuristic config 'params' must be an object.")
    return create_heuristic(heuristic_key, params)
