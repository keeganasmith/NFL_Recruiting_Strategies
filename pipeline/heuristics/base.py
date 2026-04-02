from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class Heuristic(Protocol):
    """Interface for all heuristic implementations."""

    def name(self) -> str:
        """Stable heuristic name for logging/reporting."""

    def required_columns(self) -> list[str]:
        """Columns this heuristic requires to score rows."""

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Return a per-player score indexed like `df`."""

    def metadata(self) -> dict[str, Any]:
        """Optional metadata for manifests/reporting."""
        return {}
