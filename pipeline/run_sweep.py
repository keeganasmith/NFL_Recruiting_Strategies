from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.config import load_experiment_config
from pipeline.run_experiment import run_experiment


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML configs. Install with `pip install pyyaml`."
        ) from exc
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    positions = filters.get("positions")
    if positions:
        out = out[out["Pos"].isin(positions)]

    era_min = filters.get("era_min")
    if era_min is not None:
        out = out[pd.to_numeric(out["combine_year"], errors="coerce") >= int(era_min)]

    era_max = filters.get("era_max")
    if era_max is not None:
        out = out[pd.to_numeric(out["combine_year"], errors="coerce") <= int(era_max)]

    minimum_snaps = filters.get("minimum_snaps")
    if minimum_snaps is not None and "defensive_gamesPlayed" in out.columns:
        out = out[out["defensive_gamesPlayed"] >= int(minimum_snaps)]

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heuristic sweep experiments from YAML config.")
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to configs/experiments/*.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp = load_experiment_config(args.experiment_config)
    exp_path = Path(args.experiment_config)

    output_root = Path(exp.output_root) / exp.experiment_id
    output_root.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(exp.input_data, low_memory=False)
    filtered_df = _apply_filters(
        base_df,
        {
            "positions": exp.filters.positions,
            "era_min": exp.filters.era_min,
            "era_max": exp.filters.era_max,
            "minimum_snaps": exp.filters.minimum_snaps,
        },
    )
    filtered_input_path = output_root / "filtered_input.csv"
    filtered_df.to_csv(filtered_input_path, index=False)

    comparison_rows: list[dict[str, Any]] = []

    for variant in exp.variants:
        variant_id = variant["id"]
        pattern = exp.output_naming
        run_name = pattern.format(variant_id=variant_id, experiment_id=exp.experiment_id)
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        heuristic_config_path = Path(variant["heuristic_config"])
        if not heuristic_config_path.is_absolute():
            heuristic_config_path = (exp_path.parent / heuristic_config_path).resolve()

        resolved_cfg = _load_yaml(heuristic_config_path)

        manifest = run_experiment(
            input_data=str(filtered_input_path),
            heuristic_config=str(heuristic_config_path),
            output_dir=str(run_dir),
            split_seed=exp.split.seed,
            time_split_year=exp.split.time_split_year if exp.split.type == "time" else None,
            calibration_bins=exp.calibration_bins,
            resolved_config=resolved_cfg,
        )

        summary = pd.read_csv(run_dir / "score_summary.csv")
        comparison_rows.append(
            {
                "variant_id": variant_id,
                "run_directory": str(run_dir),
                "mean_score": float(summary["mean"].iloc[0]),
                "std_score": float(summary["std"].iloc[0]),
                "p90_score": float(summary["p90"].iloc[0]),
                "rows_scored": int(summary["n_scored"].iloc[0]),
                "heuristic_name": manifest["heuristic"]["name"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["p90_score", "mean_score"], ascending=False
    )
    comparison_df.to_csv(output_root / "comparison_table.csv", index=False)

    sweep_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_config_path": str(Path(args.experiment_config).resolve()),
        "output_root": str(output_root.resolve()),
        "comparison_table": "comparison_table.csv",
        "variants": [v["id"] for v in exp.variants],
    }
    (output_root / "sweep_manifest.json").write_text(json.dumps(sweep_manifest, indent=2))


if __name__ == "__main__":
    main()
