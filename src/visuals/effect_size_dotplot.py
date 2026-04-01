"""Generate position-level standardized effect size dotplots.

This module builds small-multiple dotplots from tidy model effect output with
columns:
    - position_group
    - metric
    - estimate
    - ci_low
    - ci_high

The resulting chart uses a shared x-axis, a globally consistent metric order,
uncertainty intervals, and color encoding for positive/negative marginal
effects.
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REQUIRED_COLUMNS = {"position_group", "metric", "estimate", "ci_low", "ci_high"}
POSITIVE_COLOR = "#2166AC"
NEGATIVE_COLOR = "#B2182B"


def _validate_input_columns(effects_df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(effects_df.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(
            "Input effects table is missing required columns: " f"{missing_columns}."
        )


def prepare_effects_for_plot(effects_df: pd.DataFrame) -> pd.DataFrame:
    """Return a plotting-ready DataFrame with globally consistent metric order.

    Metrics are sorted once using overall descending mean absolute effect size
    (`mean(abs(estimate))`), then alphabetically as a tie-breaker.
    """

    _validate_input_columns(effects_df)

    clean_df = effects_df.copy()
    clean_df["position_group"] = clean_df["position_group"].astype(str)
    clean_df["metric"] = clean_df["metric"].astype(str)

    clean_df["abs_estimate"] = clean_df["estimate"].abs()
    metric_order = (
        clean_df.groupby("metric", as_index=False)["abs_estimate"]
        .mean()
        .sort_values(by=["abs_estimate", "metric"], ascending=[False, True])
    )
    metric_rank = {metric: idx for idx, metric in enumerate(metric_order["metric"])}
    clean_df["metric_order"] = clean_df["metric"].map(metric_rank).astype(int)
    clean_df["effect_sign"] = (
        clean_df["estimate"].ge(0).map({True: "positive", False: "negative"})
    )

    return clean_df


def build_effect_size_dotplot(
    effects_df: pd.DataFrame,
    heuristic_version: str,
    model_version: str,
    chart_title: str = "Standardized Combine Metric Effects on NFL Production Value by Position",
    ncols: int = 3,
) -> go.Figure:
    """Construct the small-multiple effect-size dotplot figure."""

    plot_df = prepare_effects_for_plot(effects_df)
    positions: list[str] = sorted(plot_df["position_group"].unique())

    if not positions:
        raise ValueError("Input effects table is empty after validation.")

    nrows = ceil(len(positions) / ncols)
    subplot_titles = [f"Position: {position}" for position in positions]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        horizontal_spacing=0.07,
        vertical_spacing=0.10,
    )

    legend_seen: set[str] = set()
    metric_order = (
        plot_df[["metric", "metric_order"]]
        .drop_duplicates()
        .sort_values("metric_order")["metric"]
        .tolist()
    )

    for index, position in enumerate(positions):
        row = (index // ncols) + 1
        col = (index % ncols) + 1

        panel_df = plot_df.loc[plot_df["position_group"] == position].copy()
        panel_df["metric"] = pd.Categorical(
            panel_df["metric"],
            categories=metric_order,
            ordered=True,
        )
        panel_df = panel_df.sort_values("metric")

        for sign_label, sign_color in (
            ("positive", POSITIVE_COLOR),
            ("negative", NEGATIVE_COLOR),
        ):
            sign_df = panel_df.loc[panel_df["effect_sign"] == sign_label]
            if sign_df.empty:
                continue

            showlegend = sign_label not in legend_seen
            fig.add_trace(
                go.Scatter(
                    x=sign_df["estimate"],
                    y=sign_df["metric"],
                    mode="markers",
                    marker={
                        "size": 9,
                        "color": sign_color,
                        "line": {"width": 0.5, "color": "#2f2f2f"},
                    },
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": (sign_df["ci_high"] - sign_df["estimate"]).clip(
                            lower=0
                        ),
                        "arrayminus": (sign_df["estimate"] - sign_df["ci_low"]).clip(
                            lower=0
                        ),
                        "color": "#555555",
                        "thickness": 1.5,
                        "width": 0,
                    },
                    name=f"{sign_label.title()} effect",
                    legendgroup=sign_label,
                    showlegend=showlegend,
                    customdata=sign_df[["ci_low", "ci_high"]],
                    hovertemplate=(
                        "Position: " + position + "<br>"
                        "Metric: %{y}<br>"
                        "Estimate: %{x:.3f}<br>"
                        "CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            legend_seen.add(sign_label)

        fig.update_yaxes(
            row=row,
            col=col,
            title_text="Metric" if col == 1 else None,
            categoryorder="array",
            categoryarray=metric_order,
            autorange="reversed",
        )
        fig.update_xaxes(row=row, col=col, zeroline=False)

    subtitle = (
        f"Heuristic version: {heuristic_version} · " f"Model version: {model_version}"
    )
    fig.update_layout(
        title={
            "text": f"{chart_title}<br><sup>{subtitle}</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_white",
        width=420 * ncols,
        height=max(320 * nrows, 420),
        legend={
            "orientation": "h",
            "y": -0.14,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin={"l": 70, "r": 40, "t": 150, "b": 105},
    )
    fig.update_xaxes(title_text="Standardized marginal effect")

    for shape_index in range(1, len(positions) + 1):
        row = ((shape_index - 1) // ncols) + 1
        col = ((shape_index - 1) % ncols) + 1
        fig.add_vline(
            x=0,
            line_width=1.2,
            line_dash="dash",
            line_color="#777777",
            row=row,
            col=col,
        )

    return fig


def export_figure(fig: go.Figure, output_stem: Path) -> tuple[Path, Path]:
    """Export the figure to both SVG and PDF vector formats."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_stem.with_suffix(".svg")
    pdf_path = output_stem.with_suffix(".pdf")

    fig.write_image(str(svg_path))
    fig.write_image(str(pdf_path))

    return svg_path, pdf_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create position-level standardized combine effect-size dotplots."
    )
    parser.add_argument("input_csv", type=Path, help="Path to tidy effects CSV.")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("outputs/visualizations/effect_size_dotplot"),
        help="Output file stem (without extension).",
    )
    parser.add_argument(
        "--heuristic-version",
        required=True,
        help="Heuristic version label to embed in subtitle.",
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help="Model version label to embed in subtitle.",
    )
    parser.add_argument(
        "--title",
        default="Standardized Combine Metric Effects on NFL Production Value by Position",
        help="Chart title.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of subplot columns.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    effects_df = pd.read_csv(args.input_csv)
    fig = build_effect_size_dotplot(
        effects_df=effects_df,
        heuristic_version=args.heuristic_version,
        model_version=args.model_version,
        chart_title=args.title,
        ncols=args.columns,
    )
    svg_path, pdf_path = export_figure(fig, args.output_stem)
    print(f"Saved chart: {svg_path}")
    print(f"Saved chart: {pdf_path}")


if __name__ == "__main__":
    main()
