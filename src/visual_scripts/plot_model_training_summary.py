"""Create companion visuals for `docs/model_training_results_summary.md`.

This script reads model training outputs and exports three PNG figures:
1) model performance comparison
2) prediction error profile
3) global explanation signal comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    df = pd.read_csv(path)
    missing = required_columns.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required column(s): {missing_cols}")

    return df


def _slug_to_label(model_slug: str) -> str:
    return model_slug.replace("_", " ").title()


def build_performance_figure(metrics_df: pd.DataFrame) -> plt.Figure:
    trained = metrics_df.loc[metrics_df["status"] == "trained"].copy()
    if trained.empty:
        raise ValueError("No trained models found in model metrics.")

    trained = trained.sort_values("test_rmse")
    x = np.arange(len(trained))
    width = 0.38

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10.5, 8), sharex=True)

    axes[0].bar(x - width / 2, trained["test_rmse"], width=width, color="#4e79a7", label="Test RMSE")
    axes[0].bar(x + width / 2, trained["test_mae"], width=width, color="#f28e2b", label="Test MAE")
    axes[0].set_ylabel("Error")
    axes[0].set_title("Model Performance (Lower is Better)")
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, trained["test_r2"], width=0.55, color="#59a14f")
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[1].set_ylabel("Test R²")
    axes[1].set_title("Explained Variance")
    axes[1].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_slug_to_label(m) for m in trained["model"]], rotation=0)

    fig.suptitle("Explainable Model Training Summary", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    return fig


def build_error_profile_figure(prediction_frames: dict[str, pd.DataFrame]) -> plt.Figure:
    if not prediction_frames:
        raise ValueError("No prediction frames available to visualize.")

    models = sorted(prediction_frames)
    abs_errors = []
    mean_errors = []

    for model in models:
        df = prediction_frames[model].copy()
        abs_errors.append((df["predicted"] - df["actual"]).abs().to_numpy())
        mean_errors.append(float((df["predicted"] - df["actual"]).mean()))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5.6))

    box = axes[0].boxplot(
        abs_errors,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#8B0000", "linewidth": 1.8},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#a0cbe8")
        patch.set_alpha(0.75)

    axes[0].set_xticklabels([_slug_to_label(m) for m in models], rotation=12, ha="right")
    axes[0].set_ylabel("|Pred - Actual|")
    axes[0].set_title("Absolute Error Distribution")
    axes[0].grid(axis="y", alpha=0.25)

    colors = ["#e15759" if value < 0 else "#76b7b2" for value in mean_errors]
    axes[1].bar(np.arange(len(models)), mean_errors, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[1].set_xticks(np.arange(len(models)))
    axes[1].set_xticklabels([_slug_to_label(m) for m in models], rotation=12, ha="right")
    axes[1].set_ylabel("Mean Error (Pred - Actual)")
    axes[1].set_title("Bias Direction")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Prediction Error Profile", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    return fig


def _standardize_explanation_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {col.lower(): col for col in df.columns}
    feature_col = normalized.get("feature")
    if feature_col is None:
        feature_col = normalized.get("term")

    value_col = normalized.get("importance")
    if value_col is None:
        value_col = normalized.get("coefficient")
    if value_col is None:
        value_col = normalized.get("value")

    if feature_col is None or value_col is None:
        raise ValueError("Global explanation files must contain feature/term and importance/coefficient columns.")

    standardized = df[[feature_col, value_col]].copy()
    standardized.columns = ["feature", "value"]
    standardized["value"] = pd.to_numeric(standardized["value"], errors="coerce")
    standardized = standardized.dropna(subset=["value"])
    return standardized


def build_explanation_figure(explanations: dict[str, pd.DataFrame], top_n: int = 12) -> plt.Figure:
    usable: dict[str, pd.DataFrame] = {}
    for model, df in explanations.items():
        if df.empty:
            continue
        standardized = _standardize_explanation_columns(df)
        if standardized.empty:
            continue
        usable[model] = standardized

    if not usable:
        raise ValueError("No usable explanation data found.")

    top_features: set[str] = set()
    for df in usable.values():
        ranked = df.assign(abs_value=df["value"].abs()).sort_values("abs_value", ascending=False).head(top_n)
        top_features.update(ranked["feature"].tolist())

    if not top_features:
        raise ValueError("No features available after ranking explanation values.")

    matrix = pd.DataFrame(index=sorted(top_features), columns=sorted(usable.keys()), dtype=float)
    for model, df in usable.items():
        values = df.groupby("feature")["value"].mean()
        matrix[model] = matrix.index.to_series().map(values)

    filled = matrix.fillna(0.0)
    vmax = float(np.nanmax(np.abs(filled.to_numpy())))
    vmax = max(vmax, 1e-9)

    fig_height = max(4.8, 0.35 * len(filled))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    image = ax.imshow(filled.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(filled.columns)))
    ax.set_xticklabels([_slug_to_label(m) for m in filled.columns], rotation=0)
    ax.set_yticks(np.arange(len(filled.index)))
    ax.set_yticklabels(filled.index)
    ax.set_title("Global Explanation Signal Map\n(top features by |importance/coefficient|)")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Signed importance / coefficient")

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create model training summary visuals.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/model_training/model_metrics.csv"),
        help="Path to model metrics CSV.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/model_training"),
        help="Directory containing *_test_predictions.csv files.",
    )
    parser.add_argument(
        "--explanations-dir",
        type=Path,
        default=Path("outputs/model_training"),
        help="Directory containing *_global_explanations.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations/model_training_summary"),
        help="Directory to write exported figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metrics_df = _read_csv(
        args.metrics_csv,
        {"model", "status", "test_rmse", "test_mae", "test_r2"},
    )

    prediction_frames: dict[str, pd.DataFrame] = {}
    for path in sorted(args.predictions_dir.glob("*_test_predictions.csv")):
        model = path.stem.replace("_test_predictions", "")
        df = _read_csv(path, {"predicted", "actual"})
        if df.empty:
            continue
        prediction_frames[model] = df

    explanation_frames: dict[str, pd.DataFrame] = {}
    for path in sorted(args.explanations_dir.glob("*_global_explanations.csv")):
        model = path.stem.replace("_global_explanations", "")
        df = pd.read_csv(path)
        if df.empty:
            continue
        explanation_frames[model] = df

    performance_fig = build_performance_figure(metrics_df)
    save_figure(performance_fig, args.output_dir / "model_performance.png")

    error_fig = build_error_profile_figure(prediction_frames)
    save_figure(error_fig, args.output_dir / "prediction_error_profile.png")

    explanation_fig = build_explanation_figure(explanation_frames)
    save_figure(explanation_fig, args.output_dir / "global_explanation_map.png")

    print(f"Saved visuals to: {args.output_dir}")


if __name__ == "__main__":
    main()
