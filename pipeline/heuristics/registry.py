from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.config import HeuristicConfig, load_heuristic_config
from pipeline.heuristics.base import Heuristic
from pipeline.heuristics.nfl_production_value import ConfigurableNflProductionHeuristic

HEURISTIC_REGISTRY: dict[str, type] = {
    "weighted_nfl_production_value": ConfigurableNflProductionHeuristic,
}


def _from_legacy_json(path: Path) -> HeuristicConfig:
    config = json.loads(path.read_text())
    key = config.get("heuristic_key")
    params: dict[str, Any] = config.get("params", {})

    if key != "weighted_nfl_production_value":
        raise ValueError(
            f"Unknown heuristic_key '{key}'. Available: {sorted(HEURISTIC_REGISTRY)}"
        )

    return HeuristicConfig(
        heuristic_id=key,
        feature_weights={
            "offense_yards": float(params.get("offense_weight", 1.0)),
            "touchdowns": float(params.get("touchdowns_weight", 10.0)),
            "defense_impact": float(params.get("defense_weight", 1.0)),
            "special_teams_impact": float(params.get("special_teams_weight", 0.5)),
            "availability_factor": float(params.get("availability_weight", 20.0)),
        },
        thresholds={},
        role_overrides={},
    )


def build_heuristic(config_path: str | Path) -> Heuristic:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Heuristic config not found: {path}")

    if path.suffix.lower() == ".json":
        heuristic_cfg = _from_legacy_json(path)
    else:
        heuristic_cfg = load_heuristic_config(path)

    if heuristic_cfg.heuristic_id not in HEURISTIC_REGISTRY:
        raise ValueError(
            f"Unknown heuristic id '{heuristic_cfg.heuristic_id}'. Available: {sorted(HEURISTIC_REGISTRY)}"
        )

    heuristic_cls = HEURISTIC_REGISTRY[heuristic_cfg.heuristic_id]
    return heuristic_cls(config=heuristic_cfg)
