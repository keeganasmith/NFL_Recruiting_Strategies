from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from autogluon.tabular import TabularPredictor
except ImportError:  # pragma: no cover - exercised in environments without AutoGluon
    TabularPredictor = None  # type: ignore[assignment]

TARGET_COLUMN = "NFL_production_value"
SPLIT_COLUMN = "dataset_split"
EXPLICIT_IDENTITY_COLUMNS = {
    "Player",
    "NFL_id",
    "player_id",
    "record_id",
}


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

    if model_name in {"ridge", "elastic_net"}:
        values = model.coef_
        metric = "coefficient"
    elif model_name == "decision_tree":
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


def train_models(input_csv: Path, output_dir: Path) -> None:
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
    y_train_val = pd.to_numeric(train_val_df[TARGET_COLUMN], errors="coerce")
    X_test = test_df[feature_columns]
    y_test = pd.to_numeric(test_df[TARGET_COLUMN], errors="coerce")

    valid_train_val_mask = y_train_val.notna()
    valid_test_mask = y_test.notna()

    X_train_val = X_train_val.loc[valid_train_val_mask]
    y_train_val = y_train_val.loc[valid_train_val_mask]
    X_test = X_test.loc[valid_test_mask]
    y_test = y_test.loc[valid_test_mask]

    if X_train_val.empty:
        raise ValueError("No non-null targets available in train/val rows.")
    if X_test.empty:
        raise ValueError("No non-null targets available in test rows.")

    if val_df.empty:
        cv_strategy: PredefinedSplit | KFold = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        split_marker = np.where(train_val_df[SPLIT_COLUMN].astype(str).str.lower() == "val", 0, -1)
        split_marker = split_marker[valid_train_val_mask.to_numpy()]
        cv_strategy = PredefinedSplit(test_fold=split_marker)

    searches = _build_searches(X_train_val, cv_strategy=cv_strategy)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_records: list[dict[str, float | str]] = []

    for model_name, search in searches.items():
        search.fit(X_train_val, y_train_val)
        best_pipeline = search.best_estimator_
        transformed_feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        _assert_no_forbidden_features(transformed_feature_names, forbidden_columns=identity_columns)

        predictions = best_pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        metrics_records.append(
            {
                "model": model_name,
                "best_cv_rmse": float(-search.best_score_),
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2,
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

    if TabularPredictor is None:
        metrics_records.append(
            {
                "model": "autogluon_extreme",
                "best_cv_rmse": np.nan,
                "test_rmse": np.nan,
                "test_mae": np.nan,
                "test_r2": np.nan,
                "best_params": json.dumps(
                    {"status": "skipped", "reason": "autogluon.tabular is not installed"},
                    sort_keys=True,
                ),
            }
        )
        pd.DataFrame(columns=["feature", "importance", "abs_value"]).to_csv(
            output_dir / "autogluon_extreme_global_explanations.csv",
            index=False,
        )
    else:
        autogluon_path = output_dir / "autogluon_extreme_predictor"
        train_val_ag = pd.concat([X_train_val, y_train_val.rename(TARGET_COLUMN)], axis=1)
        fit_kwargs = {
            "train_data": train_val_ag,
            "presets": "extreme",
        }
        if not val_df.empty:
            val_ag = val_df[feature_columns].copy()
            val_ag[TARGET_COLUMN] = pd.to_numeric(val_df[TARGET_COLUMN], errors="coerce")
            val_ag = val_ag.loc[val_ag[TARGET_COLUMN].notna()]
            if not val_ag.empty:
                fit_kwargs["tuning_data"] = val_ag

        predictor = TabularPredictor(
            label=TARGET_COLUMN,
            path=str(autogluon_path),
            problem_type="regression",
            eval_metric="root_mean_squared_error",
        ).fit(**fit_kwargs)

        ag_predictions = predictor.predict(X_test)
        ag_rmse = float(np.sqrt(mean_squared_error(y_test, ag_predictions)))
        ag_mae = float(mean_absolute_error(y_test, ag_predictions))
        ag_r2 = float(r2_score(y_test, ag_predictions))

        ag_importance = predictor.feature_importance(train_val_ag, silent=True)
        explanation_df = _extract_autogluon_global_explanations(ag_importance)
        _assert_no_forbidden_features(
            explanation_df["feature"].astype(str).tolist(),
            forbidden_columns=identity_columns,
        )
        explanation_df.to_csv(output_dir / "autogluon_extreme_global_explanations.csv", index=False)

        metrics_records.append(
            {
                "model": "autogluon_extreme",
                "best_cv_rmse": np.nan,
                "test_rmse": ag_rmse,
                "test_mae": ag_mae,
                "test_r2": ag_r2,
                "best_params": json.dumps(
                    {
                        "model_best": predictor.model_best,
                        "presets": "extreme",
                    },
                    sort_keys=True,
                ),
            }
        )

        ag_predictions_df = test_df.loc[valid_test_mask].copy()
        ag_predictions_df["autogluon_extreme_prediction"] = ag_predictions.to_numpy()
        ag_predictions_df[
            [SPLIT_COLUMN, TARGET_COLUMN, "autogluon_extreme_prediction"]
        ].to_csv(output_dir / "autogluon_extreme_test_predictions.csv", index=False)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_models(input_csv=args.input_csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
