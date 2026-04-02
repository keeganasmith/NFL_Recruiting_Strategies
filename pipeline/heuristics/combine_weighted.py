from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CombineWeightedHeuristic:
    """Simple weighted combine-score heuristic driven by config."""

    weights: dict[str, float] = field(default_factory=dict)
    invert_columns: list[str] = field(default_factory=list)

    def name(self) -> str:
        return "combine_weighted"

    def required_columns(self) -> list[str]:
        return list(self.weights.keys())

    def metadata(self) -> dict[str, object]:
        return {
            "weights": self.weights,
            "invert_columns": self.invert_columns,
        }

    def score(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index, dtype=float)
        for col, weight in self.weights.items():
            values = pd.to_numeric(df[col], errors="coerce")
            if col in self.invert_columns:
                values = -values
            centered = values - values.mean(skipna=True)
            scale = centered.std(skipna=True)
            z = centered / scale if scale and scale > 0 else centered * 0.0
            score = score + z.fillna(0.0) * float(weight)
        return score
