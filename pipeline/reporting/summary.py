from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_summary_outputs(
    output_dir: str | Path,
    scored_df: pd.DataFrame,
    metrics: dict[str, Any],
    calibration_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    """Persist standard experiment artifacts to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scored_df.to_csv(out / "scored_players.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    calibration_df.to_csv(out / "calibration.csv", index=False)
    ranking_df.to_csv(out / "ranking_topk.csv", index=False)
