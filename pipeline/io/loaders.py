from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"Player", "Pos", "combine_year"}


def load_all_data(path: str | Path) -> pd.DataFrame:
    """Load the all_data CSV with low-memory disabled for stable dtypes."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")
    return pd.read_csv(data_path, low_memory=False)


def validate_all_data(df: pd.DataFrame) -> None:
    """Validate minimum schema for the production-value pipeline."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Input data is missing required columns for pipeline execution: "
            f"{missing}"
        )
    if df.empty:
        raise ValueError("Input dataframe is empty.")
