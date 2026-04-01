"""Build one-page model diagnostics charts for static reporting."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class DiagnosticsThresholds:
    max_mae: float = 0.75
    max_abs_calibration_intercept: float = 0.10
    max_abs_calibration_slope_deviation: float = 0.15
    max_abs_mean_residual: float = 0.12

    @classmethod
    def from_dict(cls, raw: dict | None) -> "DiagnosticsThresholds":
        if not raw:
            return cls()
        return cls(
            max_mae=float(raw.get("max_mae", cls.max_mae)),
            max_abs_calibration_intercept=float(
                raw.get(
                    "max_abs_calibration_intercept", cls.max_abs_calibration_intercept
                )
            ),
            max_abs_calibration_slope_deviation=float(
                raw.get(
                    "max_abs_calibration_slope_deviation",
                    cls.max_abs_calibration_slope_deviation,
                )
            ),
            max_abs_mean_residual=float(
                raw.get("max_abs_mean_residual", cls.max_abs_mean_residual)
            ),
        )


def _load_thresholds(config_path: Path | None) -> DiagnosticsThresholds:
    if config_path is None or not config_path.exists():
        return DiagnosticsThresholds()
    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return DiagnosticsThresholds.from_dict(raw.get("diagnostics_thresholds", raw))


def _calibration_by_position(predictions: pd.DataFrame, bins: int) -> pd.DataFrame:
    recs: list[dict] = []
    for position, group in predictions.groupby("position_group"):
        panel = (
            group[["predicted_production_value", "target_production_value"]]
            .dropna()
            .copy()
        )
        q = min(bins, panel["predicted_production_value"].nunique())
        if panel.empty or q < 2:
            continue
        panel["bin"] = pd.qcut(
            panel["predicted_production_value"], q=q, duplicates="drop"
        )
        for interval, b in panel.groupby("bin", observed=False):
            n = len(b)
            if n == 0:
                continue
            obs = b["target_production_value"].mean()
            se = b["target_production_value"].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            ci = 1.96 * se
            recs.append(
                {
                    "position_group": position,
                    "bin": str(interval),
                    "predicted_mean": b["predicted_production_value"].mean(),
                    "actual_mean": obs,
                    "actual_ci_low": obs - ci,
                    "actual_ci_high": obs + ci,
                    "n": n,
                }
            )
    return pd.DataFrame(recs)


def _fit_calibration_line(df: pd.DataFrame) -> tuple[float, float]:
    data = df[["predicted_production_value", "target_production_value"]].dropna()
    if len(data) < 2:
        return 1.0, 0.0
    slope, intercept = np.polyfit(
        data["predicted_production_value"], data["target_production_value"], 1
    )
    return float(slope), float(intercept)


def _build_position_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, g in predictions.groupby("position_group"):
        panel = g[
            ["predicted_production_value", "target_production_value", "residual"]
        ].dropna()
        if panel.empty:
            continue
        slope, intercept = _fit_calibration_line(panel)
        rows.append(
            {
                "position_group": position,
                "n": len(panel),
                "mae": panel["residual"].abs().mean(),
                "mean_residual": panel["residual"].mean(),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
        )
    return pd.DataFrame(rows).sort_values("position_group")


def _warning_messages(
    metrics: pd.DataFrame, thresholds: DiagnosticsThresholds
) -> list[str]:
    warnings: list[str] = []
    for _, row in metrics.iterrows():
        issues = []
        if row["mae"] > thresholds.max_mae:
            issues.append(f"MAE {row['mae']:.2f}>{thresholds.max_mae:.2f}")
        if abs(row["calibration_intercept"]) > thresholds.max_abs_calibration_intercept:
            issues.append(
                f"|int| {abs(row['calibration_intercept']):.2f}>{thresholds.max_abs_calibration_intercept:.2f}"
            )
        slope_dev = abs(row["calibration_slope"] - 1.0)
        if slope_dev > thresholds.max_abs_calibration_slope_deviation:
            issues.append(
                f"|slope-1| {slope_dev:.2f}>{thresholds.max_abs_calibration_slope_deviation:.2f}"
            )
        if abs(row["mean_residual"]) > thresholds.max_abs_mean_residual:
            issues.append(
                f"|mean resid| {abs(row['mean_residual']):.2f}>{thresholds.max_abs_mean_residual:.2f}"
            )
        if issues:
            warnings.append(f"{row['position_group']}: " + "; ".join(issues))
    return warnings


def build_model_diagnostics_figure(
    predictions: pd.DataFrame,
    thresholds: DiagnosticsThresholds,
    calibration_bins: int = 8,
    title: str = "Model Diagnostics by Position",
) -> tuple[go.Figure, pd.DataFrame, list[str]]:
    required = {
        "position_group",
        "target_production_value",
        "predicted_production_value",
        "residual",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    positions = sorted(predictions["position_group"].dropna().astype(str).unique())
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#393b79",
    ]
    color_map = {p: colors[i % len(colors)] for i, p in enumerate(positions)}

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "Calibration by Position (95% CI)",
            "Residual by Position",
            "MAE by Position",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )

    cal = _calibration_by_position(predictions, calibration_bins)
    for p in positions:
        pcal = cal[cal["position_group"] == p].sort_values("predicted_mean")
        if pcal.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pcal["predicted_mean"],
                y=pcal["actual_mean"],
                mode="lines+markers",
                name=p,
                legendgroup=p,
                marker={"size": 6, "color": color_map[p]},
                line={"color": color_map[p], "width": 1.6},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (pcal["actual_ci_high"] - pcal["actual_mean"]).to_numpy(),
                    "arrayminus": (
                        pcal["actual_mean"] - pcal["actual_ci_low"]
                    ).to_numpy(),
                    "thickness": 1,
                },
            ),
            row=1,
            col=1,
        )

    vals = predictions["predicted_production_value"].dropna()
    if vals.empty:
        lo, hi = (0.0, 1.0)
    else:
        span = float(vals.max() - vals.min())
        pad = max(0.1, span * 0.08)
        lo = float(vals.min() - pad)
        hi = float(vals.max() + pad)
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dash", "color": "#444"},
            name="Perfect calibration",
        ),
        row=1,
        col=1,
    )

    for p in positions:
        vals = predictions.loc[predictions["position_group"] == p, "residual"].dropna()
        fig.add_trace(
            go.Box(
                y=vals,
                name=p,
                marker_color=color_map[p],
                boxmean=True,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#444", row=2, col=1)

    metrics = _build_position_metrics(predictions)
    warnings = _warning_messages(metrics, thresholds)

    mae_sorted = metrics.sort_values("mae", ascending=False)
    fig.add_trace(
        go.Bar(
            x=mae_sorted["position_group"],
            y=mae_sorted["mae"],
            marker_color=[color_map[p] for p in mae_sorted["position_group"]],
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title={"text": title, "x": 0.02},
        template="plotly_white",
        # Export with a more compact landscape aspect ratio so the panel
        # remains readable when embedded into static reports.
        width=1800,
        height=1300,
        font={"size": 18},
        legend={"orientation": "h", "y": 1.06, "x": 0.01},
        margin={"l": 60, "r": 30, "t": 110, "b": 60},
    )

    fig.update_annotations(font={"size": 20})

    fig.update_xaxes(
        title_text="Predicted (bin mean)",
        title_font={"size": 22},
        tickfont={"size": 16},
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Observed (bin mean)",
        title_font={"size": 22},
        tickfont={"size": 16},
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Residual",
        title_font={"size": 22},
        tickfont={"size": 16},
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="Position",
        title_font={"size": 22},
        tickfont={"size": 15},
        tickangle=-40,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="Position",
        title_font={"size": 22},
        tickfont={"size": 15},
        tickangle=-40,
        row=2,
        col=2,
    )
    fig.update_yaxes(
        title_text="MAE", title_font={"size": 22}, tickfont={"size": 16}, row=2, col=2
    )
    fig.update_xaxes(range=[2.2, 4.4], row=1, col=1)
    fig.update_yaxes(range=[lo, hi], row=1, col=1)

    return fig, metrics, warnings


def export_diagnostics_assets(
    fig: go.Figure, metrics: pd.DataFrame, warnings: list[str], output_stem: Path
) -> dict[str, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    pdf = output_stem.with_suffix(".pdf")
    metrics_csv = output_stem.with_name(output_stem.name + "_metrics.csv")
    warnings_txt = output_stem.with_name(output_stem.name + "_warnings.txt")

    fig.write_image(str(png), scale=2)
    fig.write_image(str(svg))
    fig.write_image(str(pdf))
    metrics.to_csv(metrics_csv, index=False)
    warnings_txt.write_text(
        "\n".join(warnings) if warnings else "No threshold warnings detected.\n",
        encoding="utf-8",
    )

    return {
        "png": png,
        "svg": svg,
        "pdf": pdf,
        "metrics_csv": metrics_csv,
        "warnings_txt": warnings_txt,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one-page diagnostics plot for model reporting."
    )
    parser.add_argument(
        "--predictions-csv", type=Path, default=Path("outputs/modeling/predictions.csv")
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("outputs/visualizations/model_diagnostics"),
    )
    parser.add_argument(
        "--threshold-config",
        type=Path,
        default=Path("src/visuals/config/diagnostics_thresholds.json"),
    )
    parser.add_argument("--calibration-bins", type=int, default=8)
    parser.add_argument("--title", default="Model Diagnostics by Position")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    predictions = pd.read_csv(args.predictions_csv)
    thresholds = _load_thresholds(args.threshold_config)
    fig, metrics, warnings = build_model_diagnostics_figure(
        predictions, thresholds, args.calibration_bins, args.title
    )
    outputs = export_diagnostics_assets(fig, metrics, warnings, args.output_stem)
    for key, path in outputs.items():
        print(f"Saved {key}: {path}")


if __name__ == "__main__":
    main()
