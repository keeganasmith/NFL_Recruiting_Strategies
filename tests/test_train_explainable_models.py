import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.modeling.train_explainable_models import (
    _assert_no_forbidden_features,
    _build_calibration_partition,
    train_models,
)


class TrainExplainableModelsIdentityTests(unittest.TestCase):
    def test_assert_no_forbidden_features_raises_on_leakage(self):
        with self.assertRaisesRegex(ValueError, "Forbidden identity features"):
            _assert_no_forbidden_features(
                transformed_feature_names=["cat__Player_DB One", "num__forty_yard_sec_z"],
                forbidden_columns=["Player", "NFL_id"],
            )

    def test_training_excludes_identity_columns_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "input.csv"
            output_dir = tmp / "outputs"

            df = pd.DataFrame(
                {
                    "Player": [f"P{i}" for i in range(15)],
                    "NFL_id": list(range(100, 115)),
                    "record_id": list(range(200, 215)),
                    "college_name": ["School A", "School B", "School C"] * 5,
                    "cat_college_name_school_a": [1, 0, 0] * 5,
                    "dataset_split": ["train"] * 9 + ["val"] * 3 + ["test"] * 3,
                    "NFL_production_value": [float(i) for i in range(15)],
                    "forty_yard_sec_z": [0.1 * i for i in range(15)],
                    "bench_reps_z": [1.0 + 0.2 * i for i in range(15)],
                    "forty_yard_sec_missing": [0, 1, 0] * 5,
                }
            )
            df.to_csv(input_csv, index=False)

            with mock.patch("src.modeling.train_explainable_models.TabularPredictor", None):
                train_models(input_csv=input_csv, output_dir=output_dir)

            manifest = json.loads((output_dir / "training_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["excluded_identity_columns"], ["NFL_id", "Player", "record_id"])
            self.assertNotIn("Player", manifest["model_feature_columns"])
            self.assertNotIn("NFL_id", manifest["model_feature_columns"])
            self.assertNotIn("record_id", manifest["model_feature_columns"])
            self.assertEqual(manifest["excluded_non_predictive_columns"], ["cat_college_name_school_a", "college_name"])
            self.assertNotIn("college_name", manifest["model_feature_columns"])
            self.assertNotIn("cat_college_name_school_a", manifest["model_feature_columns"])

            for model_name in ["ridge", "elastic_net", "decision_tree", "autogluon_extreme"]:
                explanation = pd.read_csv(output_dir / f"{model_name}_global_explanations.csv")
                if not explanation.empty:
                    features_lower = explanation["feature"].astype(str).str.lower()
                    self.assertFalse(features_lower.str.contains("player").any())
                    self.assertFalse(features_lower.str.contains("nfl_id").any())
                    self.assertFalse(features_lower.str.contains("record_id").any())
                    self.assertFalse(features_lower.str.contains("college_name").any())
                    self.assertFalse(features_lower.str.contains("cat_college_name").any())

            for model_name in ["ridge", "elastic_net", "decision_tree"]:
                self.assertTrue((Path("visuals/outputs") / f"predicted_vs_expected_{model_name}.svg").exists())

            autogluon_predictions = output_dir / "autogluon_extreme_test_predictions.csv"
            autogluon_plot = Path("visuals/outputs") / "predicted_vs_expected_autogluon_extreme.svg"
            if autogluon_predictions.exists():
                self.assertTrue(autogluon_plot.exists())


class _DummyPreprocessor:
    def get_feature_names_out(self):
        return np.array(["num__feature_1", "num__feature_2"])


class _DummyModel:
    def __init__(self):
        self.coef_ = np.array([0.1, 0.2])


class _DummyPipeline:
    def __init__(self):
        self.named_steps = {"preprocessor": _DummyPreprocessor(), "model": _DummyModel()}
        self._baseline = 0.0

    def fit(self, X, y, model__sample_weight=None):
        if model__sample_weight is None:
            raise AssertionError("Expected sample weights during fit.")
        self._baseline = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X):
        return np.full(len(X), self._baseline, dtype=float)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class _FakeSearch:
    def __init__(self):
        self.best_estimator_ = _DummyPipeline()
        self.best_score_ = -1.0
        self.best_params_ = {"model__alpha": 1.0}

    def fit(self, X, y, model__sample_weight=None):
        if (X["feature_1"] >= 900).any():
            raise AssertionError("Tuning data should never include test rows.")
        self.best_estimator_.fit(X, y, model__sample_weight=model__sample_weight)
        return self


class TrainExplainableModelsSplitSafetyTests(unittest.TestCase):
    def _run_with_patches(self, df: pd.DataFrame) -> tuple[dict, Path]:
        tmp = Path(tempfile.mkdtemp())
        input_csv = tmp / "input.csv"
        output_dir = tmp / "outputs"
        df.to_csv(input_csv, index=False)

        calibrator_calls: list[np.ndarray] = []

        def _record_calibrator(val_predictions, val_actuals):
            calibrator_calls.append(np.asarray(val_actuals, dtype=float))
            return "mean_bias", 0.0

        with mock.patch(
            "src.modeling.train_explainable_models._build_searches",
            return_value={"ridge": _FakeSearch()},
        ), mock.patch(
            "src.modeling.train_explainable_models._fit_calibrator",
            side_effect=_record_calibrator,
        ), mock.patch(
            "src.modeling.train_explainable_models.TabularPredictor",
            None,
        ):
            train_models(input_csv=input_csv, output_dir=output_dir)

        metrics = pd.read_csv(output_dir / "model_metrics.csv")
        return {"calibrator_calls": calibrator_calls, "metrics": metrics}, output_dir

    def test_test_rows_never_used_in_tuning_or_calibration(self):
        df = pd.DataFrame(
            {
                "dataset_split": ["train"] * 12 + ["val"] * 4 + ["test"] * 3,
                "NFL_production_value": [float(i) for i in range(16)] + [1000.0, 1001.0, 1002.0],
                "feature_1": [float(i) for i in range(16)] + [999.0, 998.0, 997.0],
                "feature_2": [1.0] * 19,
            }
        )
        artifacts, _ = self._run_with_patches(df)
        self.assertTrue(artifacts["calibrator_calls"])
        observed_actuals = np.concatenate(artifacts["calibrator_calls"])
        self.assertTrue((observed_actuals < 900).all())

    def test_pipeline_works_without_val_split(self):
        df = pd.DataFrame(
            {
                "dataset_split": ["train"] * 10 + ["test"] * 3,
                "NFL_production_value": [float(i) for i in range(10)] + [50.0, 51.0, 52.0],
                "feature_1": [float(i) for i in range(13)],
                "feature_2": [1.0] * 13,
            }
        )
        artifacts, output_dir = self._run_with_patches(df)
        self.assertIn("ridge", set(artifacts["metrics"]["model"]))
        self.assertTrue((output_dir / "ridge_test_predictions.csv").exists())

    def test_calibration_partition_stable_for_small_samples(self):
        fit_indices, calibration_indices = _build_calibration_partition(n_samples=6, random_state=42)
        self.assertEqual(len(calibration_indices), 0)
        self.assertEqual(len(fit_indices), 6)

        fit_indices_tiny, calibration_indices_tiny = _build_calibration_partition(n_samples=4, random_state=42)
        self.assertEqual(len(calibration_indices_tiny), 0)
        self.assertEqual(len(fit_indices_tiny), 4)


if __name__ == "__main__":
    unittest.main()
