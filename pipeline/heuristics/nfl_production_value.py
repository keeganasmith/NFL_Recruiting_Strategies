from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WeightedNflProductionHeuristic:
    offense_weight: float = 1.0
    touchdowns_weight: float = 10.0
    defense_weight: float = 1.0
    special_teams_weight: float = 0.5
    availability_weight: float = 20.0

    def name(self) -> str:
        return "weighted_nfl_production_value"

    def required_columns(self) -> set[str]:
        return {
            "offense_yards",
            "touchdowns",
            "defense_impact",
            "special_teams_impact",
            "availability_factor",
        }

    def score(self, df: pd.DataFrame) -> pd.Series:
        missing = sorted(self.required_columns() - set(df.columns))
        if missing:
            raise ValueError(f"Heuristic input missing required columns: {missing}")

        raw_score = (
            self.offense_weight * df["offense_yards"]
            + self.touchdowns_weight * df["touchdowns"]
            + self.defense_weight * df["defense_impact"]
            + self.special_teams_weight * df["special_teams_impact"]
            + self.availability_weight * df["availability_factor"]
        )
        return raw_score.clip(lower=0.0)

    def metadata(self) -> dict[str, float | str]:
        return {
            "heuristic": self.name(),
            "offense_weight": self.offense_weight,
            "touchdowns_weight": self.touchdowns_weight,
            "defense_weight": self.defense_weight,
            "special_teams_weight": self.special_teams_weight,
            "availability_weight": self.availability_weight,
        }
