from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from pipeline.config import BinRule, HeuristicConfig, REQUIRED_FEATURES, validate_heuristic_inputs


@dataclass(frozen=True)
class ConfigurableNflProductionHeuristic:
    config: HeuristicConfig

    def name(self) -> str:
        return self.config.heuristic_id

    def required_columns(self) -> set[str]:
        return set(REQUIRED_FEATURES)

    def _role_for_row(self, row: pd.Series) -> str | None:
        pos = str(row.get("Pos", "")).upper().strip()
        role = str(row.get("role", "")).lower().strip()

        if pos == "CB":
            return "CB"
        if pos in {"S", "SS", "FS"}:
            return "S"
        if role in {"slot", "outside"}:
            return role
        return None

    @staticmethod
    def _bin_adjustment(value: float, bins: list[BinRule]) -> float:
        for rule in bins:
            min_ok = rule.min is None or value >= rule.min
            max_ok = rule.max is None or value < rule.max
            if min_ok and max_ok:
                return rule.value
        return 0.0

    def _weights_for_role(self, role: str | None) -> dict[str, float]:
        weights = dict(self.config.feature_weights)
        if role and role in self.config.role_overrides:
            weights.update(self.config.role_overrides[role].get("feature_weights", {}))
        return weights

    def _thresholds_for_role(self, role: str | None) -> dict[str, list[BinRule]]:
        thresholds = dict(self.config.thresholds)
        if role and role in self.config.role_overrides:
            thresholds.update(self.config.role_overrides[role].get("thresholds", {}))
        return thresholds

    def score(self, df: pd.DataFrame) -> pd.Series:
        validate_heuristic_inputs(df, self.config)

        scores: list[float] = []
        for _, row in df.iterrows():
            role = self._role_for_row(row)
            weights = self._weights_for_role(role)
            thresholds = self._thresholds_for_role(role)

            total = 0.0
            for feature, weight in weights.items():
                value = float(pd.to_numeric(row.get(feature), errors="coerce"))
                if pd.isna(value):
                    value = 0.0
                total += weight * value
                if feature in thresholds:
                    total += self._bin_adjustment(value, thresholds[feature])

            scores.append(max(total, 0.0))

        return pd.Series(scores, index=df.index, dtype=float)

    def metadata(self) -> dict[str, Any]:
        return {
            "heuristic": self.name(),
            "feature_weights": self.config.feature_weights,
            "thresholds": {
                k: [rule.__dict__ for rule in v] for k, v in self.config.thresholds.items()
            },
            "role_overrides": self.config.role_overrides,
        }
