"""Modeling workflow for combine-to-production value prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.scoring.production_value import (
    compute_production_value_batch,
    load_production_value_config,
)

COMBINE_FEATURES = [
    "40yd",
    "Vertical",
    "Bench",
    "Broad Jump",
    "3Cone",
    "Shuttle",
    "Ht",
    "Wt",
]
TIME_DRILLS = {"40yd", "3Cone", "Shuttle"}

POSITION_GROUP_MAP = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "OT": "OL",
    "OG": "OL",
    "C": "OL",
    "T": "OL",
    "G": "OL",
    "DE": "EDGE",
    "EDGE": "EDGE",
    "OLB": "EDGE",
    "DT": "IDL",
    "NT": "IDL",
    "DL": "IDL",
    "ILB": "LB",
    "MLB": "LB",
    "LB": "LB",
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "DB": "DB",
    "K": "ST",
    "P": "ST",
    "LS": "ST",
}


@dataclass
class PositionModelingConfig:
    model_version: str = "v1.0.0"
    bootstrap_iterations: int = 200
    min_group_size: int = 30
    train_start_year: int | None = None
    train_end_year: int | None = None
    calibration_bins: int = 10
    ridge_alpha: float = 1.0


class PositionModelingWorkflow:
    """End-to-end training pipeline for position models and pooled baseline."""

    def __init__(
        self,
        scoring_config_path: str | Path | None = None,
        config: PositionModelingConfig | None = None,
    ) -> None:
        self.scoring_config = load_production_value_config(scoring_config_path)
        self.heuristic_version = str(self.scoring_config["version"])
        self.config = config or PositionModelingConfig()

    @staticmethod
    def _normalize_height_to_inches(height_series: pd.Series) -> pd.Series:
        parsed = height_series.astype(str).str.extract(r"(?P<feet>\d+)-(?P<inches>\d+)")
        out = pd.to_numeric(parsed["feet"], errors="coerce") * 12 + pd.to_numeric(
            parsed["inches"], errors="coerce"
        )
        fallback = pd.to_numeric(height_series, errors="coerce")
        return out.fillna(fallback)

    def _position_group(self, pos_series: pd.Series) -> pd.Series:
        return pos_series.astype(str).map(POSITION_GROUP_MAP).fillna("OTHER")

    def _prepare_target(self, df: pd.DataFrame) -> pd.Series:
        if "production_value" in df.columns:
            return pd.to_numeric(df["production_value"], errors="coerce")

        required = list(self.scoring_config["components"].keys()) + [
            self.scoring_config["time_decay"]["career_year_column"],
            "Pos",
        ]
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(
                "Input dataframe must include either 'production_value' or scoring inputs. "
                f"Missing: {missing}"
            )
        scored = compute_production_value_batch(df.copy(), self.scoring_config)
        return pd.to_numeric(scored["production_value"], errors="coerce")

    def _preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        working = df.copy()
        if "combine_year" in working.columns:
            years = pd.to_numeric(working["combine_year"], errors="coerce")
            if self.config.train_start_year is not None:
                working = working.loc[years >= self.config.train_start_year]
            if self.config.train_end_year is not None:
                working = working.loc[years <= self.config.train_end_year]

        working["position_group"] = self._position_group(working["Pos"])
        working["Ht"] = self._normalize_height_to_inches(working["Ht"])
        for col in COMBINE_FEATURES:
            if col not in working.columns:
                working[col] = np.nan
            working[col] = pd.to_numeric(working[col], errors="coerce")
            if col in TIME_DRILLS:
                working[col] = -1 * working[col]

        working["target_production_value"] = self._prepare_target(working)
        working = working.dropna(subset=["target_production_value", "Pos"])

        # Missingness indicators and position-cohort standardization
        prep_meta: dict[str, Any] = {"imputation": {}, "standardization": {}}
        for col in COMBINE_FEATURES:
            miss_col = f"{col}_missing"
            working[miss_col] = working[col].isna().astype(int)

            group_med = working.groupby("position_group")[col].transform("median")
            global_med = working[col].median()
            working[col] = working[col].fillna(group_med).fillna(global_med)

            means = working.groupby("position_group")[col].transform("mean")
            stds = (
                working.groupby("position_group")[col]
                .transform("std")
                .replace(0, np.nan)
            )
            working[f"{col}_z"] = ((working[col] - means) / stds).fillna(0.0)

            prep_meta["imputation"][col] = {
                "global_median": float(global_med) if pd.notna(global_med) else None,
                "position_medians": {
                    k: (float(v) if pd.notna(v) else None)
                    for k, v in working.groupby("position_group")[col]
                    .median()
                    .to_dict()
                    .items()
                },
            }
            prep_meta["standardization"][col] = {
                k: {
                    "mean": float(v["mean"]) if pd.notna(v["mean"]) else None,
                    "std": float(v["std"]) if pd.notna(v["std"]) else None,
                }
                for k, v in working.groupby("position_group")[col]
                .agg(["mean", "std"])
                .to_dict("index")
                .items()
            }

        return working, prep_meta

    @staticmethod
    def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        eye = np.eye(X.shape[1])
        eye[0, 0] = 0.0  # do not penalize intercept
        return np.linalg.pinv(X.T @ X + alpha * eye) @ (X.T @ y)

    def _bootstrap_coefs(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        coefs = []
        n = len(y)
        for _ in range(self.config.bootstrap_iterations):
            idx = rng.integers(0, n, n)
            coefs.append(self._fit_ridge(X[idx], y[idx], self.config.ridge_alpha))
        return np.vstack(coefs)

    def _feature_cols(self) -> list[str]:
        z = [f"{c}_z" for c in COMBINE_FEATURES]
        m = [f"{c}_missing" for c in COMBINE_FEATURES]
        return z + m

    def _build_model_outputs(
        self, fitted_df: pd.DataFrame, model_rows: list[dict[str, Any]]
    ) -> dict[str, pd.DataFrame]:
        feat_df = pd.DataFrame(model_rows)

        predictions = fitted_df[
            [
                c
                for c in [
                    "Player",
                    "NFL_id",
                    "Pos",
                    "position_group",
                    "combine_year",
                    "target_production_value",
                ]
                if c in fitted_df.columns
            ]
        ].copy()
        predictions["predicted_production_value"] = fitted_df[
            "predicted_production_value"
        ]
        predictions["residual"] = (
            fitted_df["target_production_value"]
            - fitted_df["predicted_production_value"]
        )
        predictions["baseline_prediction"] = fitted_df["baseline_prediction"]
        predictions["baseline_residual"] = (
            fitted_df["target_production_value"] - fitted_df["baseline_prediction"]
        )
        predictions["pred_interval_lower"] = fitted_df["pred_interval_lower"]
        predictions["pred_interval_upper"] = fitted_df["pred_interval_upper"]
        predictions["heuristic_version"] = self.heuristic_version
        predictions["model_version"] = self.config.model_version

        diag = (
            predictions.groupby("position_group", dropna=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "n": len(g),
                        "mae": np.abs(g["residual"]).mean(),
                        "rmse": np.sqrt(np.mean(np.square(g["residual"]))),
                        "baseline_mae": np.abs(g["baseline_residual"]).mean(),
                        "baseline_rmse": np.sqrt(
                            np.mean(np.square(g["baseline_residual"]))
                        ),
                    }
                )
            )
            .reset_index()
        )
        diag["heuristic_version"] = self.heuristic_version
        diag["model_version"] = self.config.model_version

        cal = predictions.copy()
        cal["calibration_bin"] = pd.qcut(
            cal["predicted_production_value"],
            q=min(
                self.config.calibration_bins,
                cal["predicted_production_value"].nunique(),
            ),
            duplicates="drop",
        )
        calibration = (
            cal.groupby(["position_group", "calibration_bin"], observed=False)
            .agg(
                n=("predicted_production_value", "size"),
                predicted_mean=("predicted_production_value", "mean"),
                actual_mean=("target_production_value", "mean"),
                residual_mean=("residual", "mean"),
            )
            .reset_index()
        )
        calibration["heuristic_version"] = self.heuristic_version
        calibration["model_version"] = self.config.model_version

        feat_df["heuristic_version"] = self.heuristic_version
        feat_df["model_version"] = self.config.model_version

        return {
            "predictions": predictions,
            "feature_effects": feat_df,
            "diagnostics": diag,
            "calibration": calibration,
        }

    def run(
        self, df: pd.DataFrame, output_dir: str | Path = "outputs/modeling"
    ) -> dict[str, pd.DataFrame]:
        working, prep_meta = self._preprocess(df)
        features = self._feature_cols()

        rng = np.random.default_rng(42)
        model_rows: list[dict[str, Any]] = []
        working["predicted_production_value"] = np.nan
        working["baseline_prediction"] = np.nan
        working["pred_interval_lower"] = np.nan
        working["pred_interval_upper"] = np.nan

        # pooled baseline
        X_pool = np.c_[np.ones(len(working)), working[features].to_numpy()]
        y_pool = working["target_production_value"].to_numpy()
        pool_coef = self._fit_ridge(X_pool, y_pool, self.config.ridge_alpha)
        pool_boot = self._bootstrap_coefs(X_pool, y_pool, rng)
        working["baseline_prediction"] = X_pool @ pool_coef

        names = ["intercept"] + features
        for i, name in enumerate(names):
            model_rows.append(
                {
                    "position_group": "POOLED",
                    "feature": name,
                    "estimate": float(pool_coef[i]),
                    "ci_lower": float(np.quantile(pool_boot[:, i], 0.025)),
                    "ci_upper": float(np.quantile(pool_boot[:, i], 0.975)),
                    "sample_size": int(len(working)),
                    "training_window": f"{self.config.train_start_year or 'min'}-{self.config.train_end_year or 'max'}",
                }
            )

        for group, group_df in working.groupby("position_group"):
            if len(group_df) < self.config.min_group_size:
                pred = np.repeat(float(np.mean(y_pool)), len(group_df))
                working.loc[group_df.index, "predicted_production_value"] = pred
                working.loc[group_df.index, "pred_interval_lower"] = pred
                working.loc[group_df.index, "pred_interval_upper"] = pred
                continue

            X = np.c_[np.ones(len(group_df)), group_df[features].to_numpy()]
            y = group_df["target_production_value"].to_numpy()
            coef = self._fit_ridge(X, y, self.config.ridge_alpha)
            boot = self._bootstrap_coefs(X, y, rng)

            yhat = X @ coef
            boot_preds = X @ boot.T
            working.loc[group_df.index, "predicted_production_value"] = yhat
            working.loc[group_df.index, "pred_interval_lower"] = np.quantile(
                boot_preds, 0.025, axis=1
            )
            working.loc[group_df.index, "pred_interval_upper"] = np.quantile(
                boot_preds, 0.975, axis=1
            )

            for i, name in enumerate(names):
                model_rows.append(
                    {
                        "position_group": group,
                        "feature": name,
                        "estimate": float(coef[i]),
                        "ci_lower": float(np.quantile(boot[:, i], 0.025)),
                        "ci_upper": float(np.quantile(boot[:, i], 0.975)),
                        "sample_size": int(len(group_df)),
                        "training_window": f"{self.config.train_start_year or 'min'}-{self.config.train_end_year or 'max'}",
                    }
                )

        outputs = self._build_model_outputs(working, model_rows)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in outputs.items():
            frame.to_csv(out_dir / f"{name}.csv", index=False)

        metadata = {
            "model_version": self.config.model_version,
            "heuristic_version": self.heuristic_version,
            "feature_set": features,
            "combine_features_raw": COMBINE_FEATURES,
            "position_groups": sorted(working["position_group"].unique().tolist()),
            "sample_size": int(len(working)),
            "training_window": {
                "start": self.config.train_start_year,
                "end": self.config.train_end_year,
            },
            "bootstrap_iterations": self.config.bootstrap_iterations,
            "ridge_alpha": self.config.ridge_alpha,
            "preprocessing": prep_meta,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        return outputs


def run_position_modeling_workflow(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs/modeling",
    config: PositionModelingConfig | None = None,
    scoring_config_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper for pipeline execution."""
    workflow = PositionModelingWorkflow(
        scoring_config_path=scoring_config_path, config=config
    )
    return workflow.run(df=df, output_dir=output_dir)
