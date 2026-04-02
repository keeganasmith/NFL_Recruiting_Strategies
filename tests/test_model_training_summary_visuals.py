import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.visual_scripts.plot_model_training_summary import (
    _prepare_prediction_frame,
    build_error_profile_figure,
    build_explanation_figure,
    build_performance_figure,
    derive_model_order,
    save_figure,
)


class ModelTrainingSummaryVisualTests(unittest.TestCase):
    def setUp(self):
        self.metrics = pd.DataFrame(
            {
                "model": ["decision_tree", "ridge", "elastic_net", "autogluon_extreme"],
                "status": ["trained", "trained", "trained", "skipped"],
                "test_rmse": [392.44, 392.56, 393.09, None],
                "test_mae": [308.02, 309.87, 311.37, None],
                "test_r2": [0.0767, 0.0761, 0.0736, None],
            }
        )

        self.predictions = {
            "decision_tree": pd.DataFrame(
                {
                    "dataset_split": ["train", "test", "test", "train"],
                    "NFL_production_value": [140, 250, 120, 300],
                    "decision_tree_prediction": [120, 220, 150, 330],
                }
            ),
            "ridge": pd.DataFrame(
                {
                    "dataset_split": ["train", "test", "test", "train"],
                    "NFL_production_value": [130, 260, 150, 290],
                    "ridge_prediction": [110, 200, 170, 320],
                }
            ),
        }

        self.explanations = {
            "decision_tree": pd.DataFrame(
                {
                    "feature": ["num__combine_year", "num__college_interceptions_z"],
                    "importance": [0.629, 0.371],
                }
            ),
            "ridge": pd.DataFrame(
                {
                    "feature": ["num__combine_year", "num__weight_lb_z", "num__college_interceptions"],
                    "coefficient": [-0.18, 0.11, 0.13],
                }
            ),
        }

    def test_builders_and_export(self):
        model_order = derive_model_order(self.metrics)
        perf_fig = build_performance_figure(self.metrics, model_order)
        prepared_predictions = {k: _prepare_prediction_frame(v) for k, v in self.predictions.items()}
        err_fig = build_error_profile_figure(prepared_predictions, model_order)
        exp_fig = build_explanation_figure(self.explanations, model_order, top_n=5)

        self.assertGreaterEqual(len(perf_fig.axes), 2)
        self.assertGreaterEqual(len(err_fig.axes), 2)
        self.assertGreaterEqual(len(exp_fig.axes), 2)  # includes colorbar axis

        with tempfile.TemporaryDirectory() as td:
            output_base = Path(td)
            save_figure(perf_fig, output_base / "model_performance.png")
            save_figure(err_fig, output_base / "prediction_error_profile.png")
            save_figure(exp_fig, output_base / "global_explanation_map.png")

            self.assertTrue((output_base / "model_performance.png").exists())
            self.assertTrue((output_base / "prediction_error_profile.png").exists())
            self.assertTrue((output_base / "global_explanation_map.png").exists())

    def test_x_axis_order_is_consistent_across_figures(self):
        model_order = derive_model_order(self.metrics)
        prepared_predictions = {k: _prepare_prediction_frame(v) for k, v in self.predictions.items()}

        perf_fig = build_performance_figure(self.metrics, model_order)
        err_fig = build_error_profile_figure(prepared_predictions, model_order)
        exp_fig = build_explanation_figure(self.explanations, model_order, top_n=5)

        perf_labels = [tick.get_text() for tick in perf_fig.axes[1].get_xticklabels()]
        err_labels = [tick.get_text() for tick in err_fig.axes[0].get_xticklabels()]
        exp_labels = [tick.get_text() for tick in exp_fig.axes[0].get_xticklabels()]

        self.assertListEqual(perf_labels, err_labels)
        self.assertListEqual(perf_labels, exp_labels)

    def test_prepare_prediction_frame_supports_repository_schema(self):
        prepared = _prepare_prediction_frame(self.predictions["decision_tree"])
        self.assertListEqual(prepared.columns.tolist(), ["predicted", "actual"])
        self.assertEqual(len(prepared), 2)


if __name__ == "__main__":
    unittest.main()
