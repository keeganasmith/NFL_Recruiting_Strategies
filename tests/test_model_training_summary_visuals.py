import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.visual_scripts.plot_model_training_summary import (
    build_error_profile_figure,
    build_explanation_figure,
    build_performance_figure,
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
                    "predicted": [120, 220, 150, 330],
                    "actual": [140, 250, 120, 300],
                }
            ),
            "ridge": pd.DataFrame(
                {
                    "predicted": [110, 200, 170, 320],
                    "actual": [130, 260, 150, 290],
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
        perf_fig = build_performance_figure(self.metrics)
        err_fig = build_error_profile_figure(self.predictions)
        exp_fig = build_explanation_figure(self.explanations, top_n=5)

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


if __name__ == "__main__":
    unittest.main()
