from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from autogluon.tabular import TabularPredictor

TARGET_COLUMN = "NFL_production_value"
SPLIT_COLUMN = "dataset_split"
EXPLICIT_IDENTITY_COLUMNS = {
    "Player",
    "NFL_id",
    "player_id",
    "record_id",
}
VISUAL_OUTPUT_DIR = Path("visuals/outputs")


def _build_preprocessor(
    X: pd.DataFrame,
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _looks_like_identity_column(column_name: str) -> bool:
    normalized = column_name.strip().lower()
    if normalized in {value.lower() for value in EXPLICIT_IDENTITY_COLUMNS}:
        return True
    return normalized.endswith("_id")


def _resolve_identity_columns(columns: list[str]) -> list[str]:
    return sorted({column for column in columns if _looks_like_identity_column(column)})


def _assert_no_forbidden_features(
    transformed_feature_names: list[str] | np.ndarray,
    forbidden_columns: list[str],
) -> None:
    if not forbidden_columns:
        return

    normalized_forbidden = {column.lower() for column in forbidden_columns}
    violations: list[str] = []
    for feature_name in transformed_feature_names:
        lowered = str(feature_name).lower()
        if any(
            lowered == forbidden
            or lowered.endswith(f"__{forbidden}")
            or lowered.startswith(f"{forbidden}_")
            or f"__{forbidden}_" in lowered
            for forbidden in normalized_forbidden
        ):
            violations.append(str(feature_name))

    if violations:
        raise ValueError(
            "Forbidden identity features detected in model matrix after preprocessing: "
            f"{sorted(set(violations))}"
        )


def _build_searches(X_train_val: pd.DataFrame, cv_strategy: PredefinedSplit | KFold) -> dict[str, GridSearchCV]:
    linear_preprocessor = _build_preprocessor(X_train_val, scale_numeric=True)
    tree_preprocessor = _build_preprocessor(X_train_val, scale_numeric=False)

    model_specs = {
        "ridge": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", linear_preprocessor),
                    ("model", Ridge(random_state=42)),
                ]
            ),
            "grid": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
        "elastic_net": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", linear_preprocessor),
                    ("model", ElasticNet(random_state=42, max_iter=10_000)),
                ]
            ),
            "grid": {
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.2, 0.5, 0.8, 1.0],
            },
        },
        "decision_tree": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    ("model", DecisionTreeRegressor(random_state=42)),
                ]
            ),
            "grid": {
                "model__max_depth": [2, 3, 4, 5, 6, 8],
                "model__min_samples_leaf": [5, 10, 20],
            },
        },
        "huber_linear": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", linear_preprocessor),
                    ("model", HuberRegressor(max_iter=1_000)),
                ]
            ),
            "grid": {
                "model__epsilon": [1.1, 1.35, 1.7, 2.0],
                "model__alpha": [0.0001, 0.001, 0.01],
            },
        },
        "gbm_huber": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    (
                        "model",
                        GradientBoostingRegressor(
                            loss="huber",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "grid": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
                "model__alpha": [0.85, 0.9, 0.95],
            },
        },
        "gbm_quantile": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    (
                        "model",
                        GradientBoostingRegressor(
                            loss="quantile",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "grid": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
                "model__alpha": [0.5, 0.8, 0.9],
            },
        },
    }

    searches = {
        name: GridSearchCV(
            estimator=spec["pipeline"],
            param_grid=spec["grid"],
            scoring="neg_root_mean_squared_error",
            cv=cv_strategy,
            n_jobs=-1,
            refit=True,
        )
        for name, spec in model_specs.items()
    }
    return searches


def _extract_global_explanations(best_pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if model_name in {"ridge", "elastic_net", "huber_linear"}:
        values = model.coef_
        metric = "coefficient"
    elif model_name in {"decision_tree", "gbm_huber", "gbm_quantile"}:
        values = model.feature_importances_
        metric = "importance"
    else:
        raise ValueError(f"Unsupported model '{model_name}' for explanation export.")

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            metric: values,
            "abs_value": np.abs(values),
        }
    ).sort_values("abs_value", ascending=False)
    return explanation


def _prepare_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_split = df[SPLIT_COLUMN].astype(str).str.lower().str.strip()
    train_df = df.loc[normalized_split == "train"].copy()
    val_df = df.loc[normalized_split == "val"].copy()
    test_df = df.loc[normalized_split == "test"].copy()

    if train_df.empty:
        raise ValueError("No rows found where dataset_split == 'train'.")
    if test_df.empty:
        raise ValueError("No rows found where dataset_split == 'test'.")

    return train_df, val_df, test_df


def _extract_autogluon_global_explanations(
    importance_df: pd.DataFrame,
) -> pd.DataFrame:
    if importance_df.empty:
        return pd.DataFrame(columns=["feature", "importance", "abs_value"])

    explanation = importance_df.copy()
    explanation.index = explanation.index.map(str)
    explanation = explanation.reset_index().rename(columns={"index": "feature"})
    if "importance" not in explanation.columns:
        numeric_columns = [column for column in explanation.columns if is_numeric_dtype(explanation[column])]
        if not numeric_columns:
            return pd.DataFrame(columns=["feature", "importance", "abs_value"])
        explanation = explanation.rename(columns={numeric_columns[0]: "importance"})
    explanation["abs_value"] = explanation["importance"].abs()
    return explanation.sort_values("abs_value", ascending=False)


def _save_predicted_vs_expected_scatterplot(
    expected_values: pd.Series | np.ndarray,
    predicted_values: pd.Series | np.ndarray,
    model_name: str,
    output_dir: Path = VISUAL_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_array = np.asarray(expected_values, dtype=float)
    predicted_array = np.asarray(predicted_values, dtype=float)
    lower_bound = float(np.nanmin([expected_array.min(), predicted_array.min()]))
    upper_bound = float(np.nanmax([expected_array.max(), predicted_array.max()]))
    span = upper_bound - lower_bound
    if span == 0:
        span = 1.0

    width = 800
    height = 600
    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    def _scale(value: float) -> tuple[float, float]:
        x = margin + ((value - lower_bound) / span) * plot_width
        y = margin + (1.0 - ((value - lower_bound) / span)) * plot_height
        return x, y

    points_svg: list[str] = []
    for expected, predicted in zip(expected_array, predicted_array):
        x = margin + ((expected - lower_bound) / span) * plot_width
        y = margin + (1.0 - ((predicted - lower_bound) / span)) * plot_height
        points_svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#1f77b4" fill-opacity="0.7" />')

    diagonal_start = _scale(lower_bound)
    diagonal_end = _scale(upper_bound)
    svg_markup = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="18" font-family="Arial">Predicted vs Expected: {model_name}</text>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222" />
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222" />
  <line x1="{diagonal_start[0]:.2f}" y1="{diagonal_start[1]:.2f}" x2="{diagonal_end[0]:.2f}" y2="{diagonal_end[1]:.2f}" stroke="#666" stroke-dasharray="6,6" />
  {''.join(points_svg)}
  <text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="13" font-family="Arial">Expected Value</text>
  <text x="18" y="{height / 2}" text-anchor="middle" font-size="13" font-family="Arial" transform="rotate(-90 18 {height / 2})">Predicted Value</text>
</svg>
"""
    (output_dir / f"predicted_vs_expected_{model_name}.svg").write_text(svg_markup, encoding="utf-8")


def _target_is_heavy_tailed(target_values: pd.Series) -> bool:
    positive_target = target_values[target_values > 0]
    if positive_target.empty:
        return False
    skewness = float(positive_target.skew())
    median = float(positive_target.median())
    if median <= 0:
        return skewness > 1.0
    p95 = float(positive_target.quantile(0.95))
    return skewness > 1.0 or (p95 / median) > 3.0


def _fit_calibrator(
    val_predictions: np.ndarray,
    val_actuals: np.ndarray,
) -> tuple[str, float | IsotonicRegression]:
    residuals = val_actuals - val_predictions
    if len(val_predictions) >= 20 and np.unique(val_predictions).size >= 5:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(val_predictions, residuals)
        return "isotonic_residual", isotonic

    mean_bias = float(np.mean(residuals))
    return "mean_bias", mean_bias


def _apply_calibration(
    predictions: np.ndarray,
    calibration_payload: float | IsotonicRegression,
    calibration_method: str,
) -> np.ndarray:
    if calibration_method == "isotonic_residual":
        return predictions + calibration_payload.predict(predictions)
    return predictions + float(calibration_payload)


def _export_diagnostics(
    model_name: str,
    output_dir: Path,
    test_frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    residual = actual - predicted
    diagnostics = pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
            "residual": residual,
            "absolute_residual": np.abs(residual),
        }
    )
    diagnostics.to_csv(output_dir / f"{model_name}_residual_vs_pred.csv", index=False)

    quantile_summary = pd.DataFrame(
        {
            "quantile": [0.05, 0.25, 0.5, 0.75, 0.95],
            "residual_quantile": np.quantile(residual, [0.05, 0.25, 0.5, 0.75, 0.95]),
            "abs_residual_quantile": np.quantile(np.abs(residual), [0.05, 0.25, 0.5, 0.75, 0.95]),
        }
    )
    quantile_summary.to_csv(output_dir / f"{model_name}_quantile_residual_summary.csv", index=False)

    subgroup_specs = {
        "position": ["position", "Position", "primary_position"],
        "cohort_year": ["cohort_year", "draft_year", "class_year", "cohort"],
    }
    subgroup_records: list[dict[str, float | str | int]] = []
    for subgroup_name, candidates in subgroup_specs.items():
        column = next((candidate for candidate in candidates if candidate in test_frame.columns), None)
        if column is None:
            continue
        subgroup_df = test_frame[[column]].copy()
        subgroup_df["actual"] = actual
        subgroup_df["predicted"] = predicted
        subgroup_df["residual"] = residual

        grouped = subgroup_df.groupby(column, dropna=False)
        for subgroup_value, chunk in grouped:
            if chunk.empty:
                continue
            subgroup_records.append(
                {
                    "subgroup_dimension": subgroup_name,
                    "subgroup_value": str(subgroup_value),
                    "n_obs": int(len(chunk)),
                    "rmse": float(np.sqrt(mean_squared_error(chunk["actual"], chunk["predicted"]))),
                    "mae": float(mean_absolute_error(chunk["actual"], chunk["predicted"])),
                    "mean_residual": float(chunk["residual"].mean()),
                }
            )

    pd.DataFrame(subgroup_records).to_csv(output_dir / f"{model_name}_subgroup_metrics.csv", index=False)


def train_models(
    input_csv: Path,
    output_dir: Path,
    autogluon_preset: str = "extreme",
) -> None:
    df = pd.read_csv(input_csv)

    required_columns = {TARGET_COLUMN, SPLIT_COLUMN}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Input data missing required columns: {missing_columns}")

    train_df, val_df, test_df = _prepare_splits(df)

    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    identity_columns = _resolve_identity_columns(train_val_df.columns.tolist())
    feature_columns = [
        column
        for column in train_val_df.columns
        if column not in {TARGET_COLUMN, SPLIT_COLUMN} and column not in identity_columns
    ]

    X_train_val = train_val_df[feature_columns]
    y_train_val_raw = pd.to_numeric(train_val_df[TARGET_COLUMN], errors="coerce")
    X_test = test_df[feature_columns]
    y_test_raw = pd.to_numeric(test_df[TARGET_COLUMN], errors="coerce")

    valid_train_val_mask = y_train_val_raw.notna()
    valid_test_mask = y_test_raw.notna()

    X_train_val = X_train_val.loc[valid_train_val_mask]
    y_train_val_raw = y_train_val_raw.loc[valid_train_val_mask]
    X_test = X_test.loc[valid_test_mask]
    y_test_raw = y_test_raw.loc[valid_test_mask]

    if X_train_val.empty:
        raise ValueError("No non-null targets available in train/val rows.")
    if X_test.empty:
        raise ValueError("No non-null targets available in test rows.")

    use_target_log1p = _target_is_heavy_tailed(y_train_val_raw)
    if use_target_log1p and (y_train_val_raw < 0).any():
        use_target_log1p = False

    if use_target_log1p:
        y_train_val = np.log1p(y_train_val_raw)
    else:
        y_train_val = y_train_val_raw.copy()

    if val_df.empty:
        cv_strategy: PredefinedSplit | KFold = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        split_marker = np.where(train_val_df[SPLIT_COLUMN].astype(str).str.lower() == "val", 0, -1)
        split_marker = split_marker[valid_train_val_mask.to_numpy()]
        cv_strategy = PredefinedSplit(test_fold=split_marker)

    searches = _build_searches(X_train_val, cv_strategy=cv_strategy)
    high_production_threshold = float(y_train_val_raw.quantile(0.75))
    train_sample_weight = np.where(y_train_val_raw >= high_production_threshold, 2.0, 1.0)
    y_test_array = y_test_raw.to_numpy(dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_records: list[dict[str, float | str]] = []

    for model_name, search in searches.items():
        search.fit(X_train_val, y_train_val, model__sample_weight=train_sample_weight)
        best_pipeline = search.best_estimator_
        transformed_feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        _assert_no_forbidden_features(transformed_feature_names, forbidden_columns=identity_columns)

        raw_predictions = best_pipeline.predict(X_test)
        if use_target_log1p:
            predictions = np.expm1(raw_predictions)
        else:
            predictions = raw_predictions

        calibration_method = "none"
        if not val_df.empty:
            val_features = val_df[feature_columns]
            val_targets_raw = pd.to_numeric(val_df[TARGET_COLUMN], errors="coerce")
            val_valid_mask = val_targets_raw.notna()
            val_features = val_features.loc[val_valid_mask]
            val_targets_raw = val_targets_raw.loc[val_valid_mask]
            if not val_features.empty:
                val_predictions_raw = best_pipeline.predict(val_features)
                if use_target_log1p:
                    val_predictions = np.expm1(val_predictions_raw)
                else:
                    val_predictions = val_predictions_raw
                calibration_method, calibration_payload = _fit_calibrator(
                    val_predictions=np.asarray(val_predictions, dtype=float),
                    val_actuals=val_targets_raw.to_numpy(dtype=float),
                )
                predictions = _apply_calibration(
                    predictions=np.asarray(predictions, dtype=float),
                    calibration_payload=calibration_payload,
                    calibration_method=calibration_method,
                )

        rmse = float(np.sqrt(mean_squared_error(y_test_array, predictions)))
        mae = float(mean_absolute_error(y_test_array, predictions))
        r2 = float(r2_score(y_test_array, predictions))

        metrics_records.append(
            {
                "model": model_name,
                "best_cv_rmse": float(-search.best_score_),
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2,
                "target_transform": "log1p" if use_target_log1p else "none",
                "calibration_method": calibration_method,
                "sample_weighting": "high_production>=p75:2x",
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
        )

        explanation_df = _extract_global_explanations(best_pipeline, model_name=model_name)
        explanation_df.to_csv(output_dir / f"{model_name}_global_explanations.csv", index=False)

        predictions_df = test_df.loc[valid_test_mask].copy()
        predictions_df[f"{model_name}_prediction"] = predictions
        predictions_df[[SPLIT_COLUMN, TARGET_COLUMN, f"{model_name}_prediction"]].to_csv(
            output_dir / f"{model_name}_test_predictions.csv", index=False
        )
        _export_diagnostics(
            model_name=model_name,
            output_dir=output_dir,
            test_frame=predictions_df,
            actual=y_test_array,
            predicted=np.asarray(predictions, dtype=float),
        )
        _save_predicted_vs_expected_scatterplot(
            expected_values=y_test_raw,
            predicted_values=predictions,
            model_name=model_name,
        )

    autogluon_preset = autogluon_preset.strip().lower()
    if autogluon_preset not in {"extreme", "medium"}:
        raise ValueError("autogluon_preset must be one of: 'extreme', 'medium'.")

    autogluon_model_name = f"autogluon_{autogluon_preset}"
    if TabularPredictor is None:
        metrics_records.append(
            {
                "model": autogluon_model_name,
                "best_cv_rmse": np.nan,
                "test_rmse": np.nan,
                "test_mae": np.nan,
                "test_r2": np.nan,
                "target_transform": "log1p" if use_target_log1p else "none",
                "calibration_method": "none",
                "sample_weighting": "not_applied",
                "best_params": json.dumps(
                    {"status": "skipped", "reason": "autogluon.tabular is not installed"},
                    sort_keys=True,
                ),
            }
        )
        pd.DataFrame(columns=["feature", "importance", "abs_value"]).to_csv(
            output_dir / f"{autogluon_model_name}_global_explanations.csv",
            index=False,
        )
    else:
        autogluon_path = output_dir / f"{autogluon_model_name}_predictor"
        train_val_ag = train_val_df.loc[valid_train_val_mask, feature_columns + [TARGET_COLUMN]]
        if use_target_log1p:
            train_val_ag = train_val_ag.copy()
            train_val_ag[TARGET_COLUMN] = np.log1p(train_val_ag[TARGET_COLUMN])

        fit_kwargs = {
            "train_data": train_val_ag,
            "presets": autogluon_preset,
        }

        predictor = TabularPredictor(
            label=TARGET_COLUMN,
            path=str(autogluon_path),
            problem_type="regression",
            eval_metric="root_mean_squared_error",
        ).fit(**fit_kwargs)

        print("predictor finished")
        ag_predictions_raw = predictor.predict(X_test)
        if use_target_log1p:
            ag_predictions = np.expm1(ag_predictions_raw.to_numpy())
        else:
            ag_predictions = ag_predictions_raw.to_numpy()
        ag_calibration_method = "none"
        if not val_df.empty:
            val_features = val_df[feature_columns]
            val_targets_raw = pd.to_numeric(val_df[TARGET_COLUMN], errors="coerce")
            val_valid_mask = val_targets_raw.notna()
            val_features = val_features.loc[val_valid_mask]
            val_targets_raw = val_targets_raw.loc[val_valid_mask]
            if not val_features.empty:
                ag_val_predictions_raw = predictor.predict(val_features).to_numpy()
                if use_target_log1p:
                    ag_val_predictions = np.expm1(ag_val_predictions_raw)
                else:
                    ag_val_predictions = ag_val_predictions_raw
                ag_calibration_method, ag_calibration_payload = _fit_calibrator(
                    val_predictions=np.asarray(ag_val_predictions, dtype=float),
                    val_actuals=val_targets_raw.to_numpy(dtype=float),
                )
                ag_predictions = _apply_calibration(
                    predictions=np.asarray(ag_predictions, dtype=float),
                    calibration_payload=ag_calibration_payload,
                    calibration_method=ag_calibration_method,
                )
        ag_rmse = float(np.sqrt(mean_squared_error(y_test_array, ag_predictions)))
        ag_mae = float(mean_absolute_error(y_test_array, ag_predictions))
        ag_r2 = float(r2_score(y_test_array, ag_predictions))

        ag_importance = predictor.feature_importance(train_val_ag, silent=True)
        explanation_df = _extract_autogluon_global_explanations(ag_importance)
        _assert_no_forbidden_features(
            explanation_df["feature"].astype(str).tolist(),
            forbidden_columns=identity_columns,
        )
        explanation_df.to_csv(output_dir / f"{autogluon_model_name}_global_explanations.csv", index=False)

        metrics_records.append(
            {
                "model": autogluon_model_name,
                "best_cv_rmse": np.nan,
                "test_rmse": ag_rmse,
                "test_mae": ag_mae,
                "test_r2": ag_r2,
                "target_transform": "log1p" if use_target_log1p else "none",
                "calibration_method": ag_calibration_method,
                "sample_weighting": "not_applied",
                "best_params": json.dumps(
                    {
                        "model_best": predictor.model_best,
                        "presets": autogluon_preset,
                    },
                    sort_keys=True,
                ),
            }
        )

        ag_predictions_df = test_df.loc[valid_test_mask].copy()
        ag_predictions_df[f"{autogluon_model_name}_prediction"] = ag_predictions
        ag_predictions_df[
            [SPLIT_COLUMN, TARGET_COLUMN, f"{autogluon_model_name}_prediction"]
        ].to_csv(output_dir / f"{autogluon_model_name}_test_predictions.csv", index=False)
        _export_diagnostics(
            model_name=autogluon_model_name,
            output_dir=output_dir,
            test_frame=ag_predictions_df,
            actual=y_test_raw.to_numpy(),
            predicted=np.asarray(ag_predictions, dtype=float),
        )
        _save_predicted_vs_expected_scatterplot(
            expected_values=y_test_raw.to_numpy(),
            predicted_values=ag_predictions,
            model_name=autogluon_model_name,
        )

    metrics_df = pd.DataFrame(metrics_records).sort_values("test_rmse")
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "target_column": TARGET_COLUMN,
                "split_column": SPLIT_COLUMN,
                "excluded_identity_columns": identity_columns,
                "model_feature_columns": feature_columns,
                "autogluon_preset": autogluon_preset,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and tune explainable regression models for NFL production prediction."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("./artifacts/db_model_preprocessed.csv"),
        help="Path to preprocessed CSV with dataset_split and NFL_production_value columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/model_training"),
        help="Directory where tuned model artifacts are written.",
    )
    parser.add_argument(
        "--autogluon-preset",
        type=str,
        choices=["medium", "extreme"],
        default="extreme",
        help=(
            "AutoGluon training quality preset. Use 'medium' for faster/lighter training "
            "or 'extreme' for highest-quality training."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_models(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        autogluon_preset=args.autogluon_preset,
    )


if __name__ == "__main__":
    main()
