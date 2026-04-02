from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.config import load_experiment_config
from pipeline.evaluation.heuristic_objective import evaluate_heuristic
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


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML configs. Install with `pip install pyyaml`."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


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


def _build_auto_grid_variants(
    *,
    experiment_config_path: Path,
    experiment_cfg: dict[str, Any],
) -> list[dict[str, str]]:
    auto_grid = experiment_cfg.get("auto_grid")
    if not auto_grid:
        return []

    base_cfg_ref = auto_grid.get("base_heuristic_config")
    if not base_cfg_ref:
        raise ValueError("auto_grid.base_heuristic_config is required when auto_grid is set.")

    base_cfg_path = Path(base_cfg_ref)
    if not base_cfg_path.is_absolute():
        base_cfg_path = (experiment_config_path.parent / base_cfg_path).resolve()

    base_cfg = _load_yaml(base_cfg_path)
    weights_grid = auto_grid.get("feature_weights", {})
    if not isinstance(weights_grid, dict) or not weights_grid:
        raise ValueError("auto_grid.feature_weights must be a non-empty mapping.")

    for feature, values in weights_grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"auto_grid.feature_weights.{feature} must be a non-empty list.")

    output_prefix = str(auto_grid.get("variant_prefix", "grid"))
    output_dir = Path(auto_grid.get("output_config_dir", "configs/heuristics"))
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    feature_names = list(weights_grid.keys())
    combos = list(product(*[weights_grid[f] for f in feature_names]))

    variants: list[dict[str, str]] = []
    for idx, combo in enumerate(combos, start=1):
        cfg = dict(base_cfg)
        cfg["feature_weights"] = dict(base_cfg.get("feature_weights", {}))
        cfg.setdefault("thresholds", base_cfg.get("thresholds", {}))
        cfg.setdefault("role_overrides", base_cfg.get("role_overrides", {}))
        for feature, value in zip(feature_names, combo):
            cfg["feature_weights"][feature] = float(value)

        variant_id = f"{output_prefix}_{idx:03d}"
        config_name = f"{variant_id}.yaml"
        config_path = output_dir / config_name
        _dump_yaml(config_path, cfg)
        try:
            config_ref = str(config_path.relative_to(PROJECT_ROOT))
        except ValueError:
            config_ref = str(config_path)
        variants.append(
            {
                "id": variant_id,
                "heuristic_config": config_ref,
            }
        )

    return variants


def _write_summary_markdown(path: Path, table: pd.DataFrame) -> None:
    cols = [
        "variant_id",
        "objective_score",
        "proxy_spearman",
        "rank_corr",
        "top_overlap",
        "calibration_rmse",
        "mean_score",
        "p90_score",
    ]
    view = table[cols].copy()
    for col in cols[1:]:
        view[col] = view[col].map(lambda v: f"{float(v):.4f}")

    header = "| " + " | ".join(cols) + " |"
    divider = "|" + "|".join(["---" for _ in cols]) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]

    lines = [
        "# Heuristic Sweep Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Top Configurations (sorted by objective_score)",
        "",
        header,
        divider,
        *rows,
        "",
        "Objective score = 0.55*proxy_spearman + 0.25*rank_corr + 0.20*top_overlap - 0.15*calibration_rmse",
    ]
    path.write_text("\n".join(lines) + "\n")


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
    exp_path = Path(args.experiment_config)
    exp = load_experiment_config(args.experiment_config)
    exp_raw = _load_yaml(exp_path)

    auto_grid_variants = _build_auto_grid_variants(
        experiment_config_path=exp_path,
        experiment_cfg=exp_raw,
    )
    variants_to_run = auto_grid_variants if auto_grid_variants else exp.variants

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

    for variant in variants_to_run:
        variant_id = variant["id"]
        pattern = exp.output_naming
        run_name = pattern.format(variant_id=variant_id, experiment_id=exp.experiment_id)
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        heuristic_config_path = Path(variant["heuristic_config"])
        if not heuristic_config_path.is_absolute():
            heuristic_config_path = (PROJECT_ROOT / heuristic_config_path).resolve()

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
        ranking = pd.read_csv(run_dir / "ranking_stability.csv")
        scored_players = pd.read_csv(run_dir / "scored_players.csv", low_memory=False)
        objective = evaluate_heuristic(scored_df=scored_players, ranking_df=ranking)
        comparison_rows.append(
            {
                "variant_id": variant_id,
                "run_directory": str(run_dir),
                "mean_score": float(summary["mean"].iloc[0]),
                "std_score": float(summary["std"].iloc[0]),
                "p90_score": float(summary["p90"].iloc[0]),
                "rows_scored": int(summary["n_scored"].iloc[0]),
                "heuristic_name": manifest["heuristic"]["name"],
                "proxy_spearman": objective.proxy_spearman,
                "calibration_rmse": objective.calibration_rmse,
                "rank_corr": objective.rank_corr,
                "top_overlap": objective.top_overlap,
                "objective_score": objective.objective_score,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["objective_score", "proxy_spearman", "p90_score"], ascending=False
    )
    comparison_df.to_csv(output_root / "comparison_table.csv", index=False)
    _write_summary_markdown(output_root / "comparison_summary.md", comparison_df.head(15))

    sweep_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_config_path": str(Path(args.experiment_config).resolve()),
        "output_root": str(output_root.resolve()),
        "comparison_table": "comparison_table.csv",
        "comparison_summary": "comparison_summary.md",
        "variants": [v["id"] for v in variants_to_run],
        "auto_grid_enabled": bool(auto_grid_variants),
    }
    (output_root / "sweep_manifest.json").write_text(json.dumps(sweep_manifest, indent=2))


if __name__ == "__main__":
    main()
