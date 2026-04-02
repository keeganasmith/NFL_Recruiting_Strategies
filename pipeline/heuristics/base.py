from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class Heuristic(Protocol):
    """Interface for NFL production-value heuristics."""

    def name(self) -> str:
        ...

    def required_columns(self) -> set[str]:
        ...

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Return per-player NFL production-value scores."""
        ...

    def metadata(self) -> dict[str, Any]:
        ...
