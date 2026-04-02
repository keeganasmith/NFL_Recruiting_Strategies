from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.evaluation.calibration import calibration_table
from pipeline.evaluation.metrics import compute_basic_metrics
from pipeline.evaluation.ranking import ranking_analysis
from pipeline.features.preprocessing import prepare_features
from pipeline.heuristics.registry import build_heuristic_from_config, load_heuristic_config
from pipeline.io.loaders import load_all_data, validate_required_columns
from pipeline.reporting.summary import write_summary_outputs


@dataclass
class SplitSettings:
    seed: int | None
    time_column: str | None
    train_end_year: int | None


def _infer_data_slice_info(df, split_settings: SplitSettings) -> dict[str, Any]:
    out: dict[str, Any] = {"n_rows": int(len(df))}
    for candidate in [split_settings.time_column, "combine_year", "season_year"]:
        if candidate and candidate in df.columns:
            years = df[candidate]
            out["time_column"] = candidate
            out["time_min"] = float(years.min()) if len(years) else None
            out["time_max"] = float(years.max()) if len(years) else None
            break
    return out


def _git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        return output or None
    except Exception:
        return None


def build_manifest(
    heuristic_name: str,
    heuristic_metadata: dict[str, Any],
    heuristic_config_path: str,
    data_path: str,
    split_settings: SplitSettings,
    evaluation_settings: dict[str, Any],
    data_slice_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "heuristic": {
            "name": heuristic_name,
            "parameters": heuristic_metadata,
            "config_path": heuristic_config_path,
        },
        "data": {
            "input_path": data_path,
            "slice": data_slice_info,
            "split": asdict(split_settings),
        },
        "evaluation": evaluation_settings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heuristic experiment pipeline.")
    parser.add_argument("--input-data", required=True, help="Path to all_data.csv input")
    parser.add_argument(
        "--heuristic-config", required=True, help="Path to heuristic config JSON"
    )
    parser.add_argument("--output-dir", required=True, help="Run output directory")
    parser.add_argument("--seed", type=int, default=None, help="Optional split seed")
    parser.add_argument(
        "--time-column",
        default=None,
        help="Optional time column name for manifest + split context",
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=None,
        help="Optional cutoff year used for time-based slicing",
    )
    parser.add_argument(
        "--target-column",
        default="scoring_totalPoints",
        help="Target column for evaluation metrics/calibration",
    )
    parser.add_argument(
        "--calibration-bins", type=int, default=10, help="Number of calibration bins"
    )
    parser.add_argument("--top-k", type=int, default=25, help="Rows in ranking output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_settings = SplitSettings(
        seed=args.seed, time_column=args.time_column, train_end_year=args.train_end_year
    )

    raw_df = load_all_data(args.input_data)
    prepared = prepare_features(raw_df)

    heuristic_config = load_heuristic_config(args.heuristic_config)
    heuristic = build_heuristic_from_config(heuristic_config)
    validate_required_columns(prepared.df, heuristic.required_columns())

    scored_df = prepared.df.copy()
    scored_df["heuristic_score"] = heuristic.score(prepared.df)

    metrics = compute_basic_metrics(
        scored_df, score_column="heuristic_score", target_column=args.target_column
    )
    calibration_df = calibration_table(
        scored_df,
        score_column="heuristic_score",
        target_column=args.target_column,
        bins=args.calibration_bins,
    )
    ranking_df = ranking_analysis(
        scored_df,
        score_column="heuristic_score",
        top_k=args.top_k,
    )

    write_summary_outputs(
        args.output_dir,
        scored_df=scored_df,
        metrics=metrics,
        calibration_df=calibration_df,
        ranking_df=ranking_df,
    )

    manifest = build_manifest(
        heuristic_name=heuristic.name(),
        heuristic_metadata=heuristic.metadata(),
        heuristic_config_path=args.heuristic_config,
        data_path=args.input_data,
        split_settings=split_settings,
        evaluation_settings={
            "target_column": args.target_column,
            "calibration_bins": args.calibration_bins,
            "top_k": args.top_k,
        },
        data_slice_info=_infer_data_slice_info(scored_df, split_settings),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
