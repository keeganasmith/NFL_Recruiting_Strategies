from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.heuristics.base import Heuristic
from pipeline.heuristics.nfl_production_value import WeightedNflProductionHeuristic

HEURISTIC_REGISTRY: dict[str, type] = {
    "weighted_nfl_production_value": WeightedNflProductionHeuristic,
}


def build_heuristic(config_path: str | Path) -> Heuristic:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Heuristic config not found: {path}")

    config = json.loads(path.read_text())
    key = config.get("heuristic_key")
    params: dict[str, Any] = config.get("params", {})

    if key not in HEURISTIC_REGISTRY:
        raise ValueError(
            f"Unknown heuristic_key '{key}'. Available: {sorted(HEURISTIC_REGISTRY)}"
        )

    heuristic_cls = HEURISTIC_REGISTRY[key]
    return heuristic_cls(**params)
