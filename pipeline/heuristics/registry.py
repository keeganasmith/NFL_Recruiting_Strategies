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
            "defensive_totalTackles": float(params.get("total_tackles_weight", 1.0)),
            "defensive_sacks": float(params.get("sacks_weight", 2.0)),
            "defensive_interceptions": float(params.get("interceptions_weight", 3.0)),
            "defensive_passesDefended": float(params.get("passes_defended_weight", 1.5)),
            "defensive_gamesPlayed": float(params.get("games_played_weight", 0.5)),
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
