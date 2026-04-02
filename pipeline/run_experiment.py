from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation.calibration import calibration_table
from pipeline.evaluation.metrics import score_summary
from pipeline.evaluation.ranking import rank_stability, top_players
from pipeline.features.preprocessing import prepare_features
from pipeline.heuristics.registry import build_heuristic
from pipeline.io.loaders import load_all_data, validate_all_data
from pipeline.reporting.plots import save_calibration_plot, save_score_distribution
from pipeline.reporting.summary import write_table


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return None


def _split_data(
    df: pd.DataFrame,
    seed: int,
    time_split_year: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | str | None]]:
    if time_split_year is not None:
        train = df[pd.to_numeric(df["combine_year"], errors="coerce") <= time_split_year]
        test = df[pd.to_numeric(df["combine_year"], errors="coerce") > time_split_year]
        split_info: dict[str, int | str | None] = {
            "split_type": "time",
            "time_split_year": time_split_year,
            "seed": None,
        }
    else:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        cut = int(len(shuffled) * 0.8)
        train, test = shuffled.iloc[:cut].copy(), shuffled.iloc[cut:].copy()
        split_info = {
            "split_type": "random",
            "time_split_year": None,
            "seed": seed,
        }
    return train, test, split_info


def run_experiment(
    *,
    input_data: str,
    heuristic_config: str,
    output_dir: str,
    split_seed: int = 42,
    time_split_year: int | None = None,
    calibration_bins: int = 10,
    resolved_config: dict | None = None,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_all_data(input_data)
    validate_all_data(raw)
    featured = prepare_features(raw)

    heuristic = build_heuristic(heuristic_config)
    featured["NFL_production_value"] = heuristic.score(featured)

    train_df, test_df, split_info = _split_data(
        featured, seed=split_seed, time_split_year=time_split_year
    )

    summary_df = score_summary(featured)
    calibration_df = calibration_table(featured, bins=calibration_bins)
    ranking_df = rank_stability(train_df, test_df)
    top_df = top_players(featured)

    write_table(summary_df, out_dir / "score_summary.csv")
    write_table(calibration_df, out_dir / "calibration_table.csv")
    write_table(ranking_df, out_dir / "ranking_stability.csv")
    write_table(top_df, out_dir / "top_players.csv")
    write_table(featured, out_dir / "scored_players.csv")

    save_score_distribution(featured, out_dir / "plots" / "score_distribution.png")
    save_calibration_plot(calibration_df, out_dir / "plots" / "calibration_curve.png")

    resolved_path = out_dir / "resolved_heuristic_config.yaml"
    if resolved_config is not None:
        try:
            import yaml  # type: ignore

            resolved_path.write_text(yaml.safe_dump(resolved_config, sort_keys=False))
        except ImportError:
            resolved_path = out_dir / "resolved_heuristic_config.json"
            resolved_path.write_text(json.dumps(resolved_config, indent=2))
    else:
        shutil.copy2(Path(heuristic_config), resolved_path)

    data_slice = {
        "rows": int(len(featured)),
        "combine_year_min": int(pd.to_numeric(featured["combine_year"], errors="coerce").min()),
        "combine_year_max": int(pd.to_numeric(featured["combine_year"], errors="coerce").max()),
        "positions": int(featured["Pos"].nunique(dropna=True)),
    }
    evaluation_settings = {
        "calibration_bins": calibration_bins,
        "ranking_top_n": 100,
        **split_info,
    }

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "heuristic": {
            "name": heuristic.name(),
            "parameters": heuristic.metadata(),
            "config_path": str(Path(heuristic_config).resolve()),
            "resolved_config_artifact": str(resolved_path.name),
        },
        "data": {
            "input_path": str(Path(input_data).resolve()),
            "slice": data_slice,
        },
        "evaluation_settings": evaluation_settings,
        "artifacts": {
            "score_summary": "score_summary.csv",
            "calibration_table": "calibration_table.csv",
            "ranking_stability": "ranking_stability.csv",
            "top_players": "top_players.csv",
            "scored_players": "scored_players.csv",
            "plots": [
                "plots/score_distribution.png",
                "plots/calibration_curve.png",
            ],
        },
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NFL production-value heuristic experiment.")
    parser.add_argument("--input-data", required=True, help="Path to all_data.csv")
    parser.add_argument("--heuristic-config", required=True, help="Path to heuristic config JSON/YAML")
    parser.add_argument("--output-dir", required=True, help="Output directory for run artifacts")
    parser.add_argument("--split-seed", type=int, default=42, help="Random split seed")
    parser.add_argument(
        "--time-split-year",
        type=int,
        default=None,
        help="Optional combine_year cutoff; <= goes to train, > goes to test",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of quantile bins used for calibration table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        input_data=args.input_data,
        heuristic_config=args.heuristic_config,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        time_split_year=args.time_split_year,
        calibration_bins=args.calibration_bins,
    )


if __name__ == "__main__":
    main()
