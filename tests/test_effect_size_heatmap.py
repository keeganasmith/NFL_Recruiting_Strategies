import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.visuals.effect_size_heatmap import build_effect_size_heatmap, export_figure
from src.visuals.effect_size_dotplot import prepare_effects_for_plot


class EffectSizeHeatmapTests(unittest.TestCase):
    def setUp(self):
        self.effects = pd.DataFrame(
            {
                "position_group": ["WR", "RB", "WR", "RB"],
                "metric": ["speed", "speed", "size", "size"],
                "estimate": [0.50, 0.10, -0.20, -0.40],
                "ci_low": [0.20, -0.10, -0.30, -0.60],
                "ci_high": [0.80, 0.30, -0.05, -0.10],
            }
        )

    def test_prepare_effects_sorts_by_global_mean_absolute_effect(self):
        prepared = prepare_effects_for_plot(self.effects)
        metric_order = (
            prepared[["metric", "metric_order"]]
            .drop_duplicates()
            .sort_values("metric_order")["metric"]
            .tolist()
        )
        self.assertEqual(metric_order, ["size", "speed"])

    def test_export_figure_writes_png_svg_pdf(self):
        fig = build_effect_size_heatmap(
            effects_df=self.effects,
            heuristic_version="h1",
            model_version="m1",
        )

        with tempfile.TemporaryDirectory() as td:
            png_path, svg_path, pdf_path = export_figure(fig, Path(td) / "heatmap")
            self.assertTrue(png_path.exists())
            self.assertTrue(svg_path.exists())
            self.assertTrue(pdf_path.exists())

    def test_heatmap_contains_only_simple_matrix(self):
        fig = build_effect_size_heatmap(
            effects_df=self.effects,
            heuristic_version="h1",
            model_version="m1",
        )

        self.assertEqual(len(fig.data), 1)
        heatmap = fig.data[0]

        self.assertEqual(list(heatmap.x), ["RB", "WR"])
        self.assertEqual(list(heatmap.y), ["size", "speed"])
        self.assertEqual(heatmap.z[0][0], -0.40)
        self.assertEqual(heatmap.z[1][1], 0.50)
        self.assertEqual(heatmap.xgap, 1)
        self.assertEqual(heatmap.ygap, 1)


if __name__ == "__main__":
    unittest.main()
