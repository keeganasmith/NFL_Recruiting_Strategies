from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class DataValidationError(ValueError):
    """Raised when required data schema checks fail."""


def load_all_data(path: str | Path) -> pd.DataFrame:
    """Load the project-level all_data CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input data was not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Assert required columns exist in the dataframe."""
    required = {col for col in required_columns if col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise DataValidationError(
            f"Input dataframe is missing required columns: {missing}"
        )
