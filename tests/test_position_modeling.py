import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.position_models import (
    PositionModelingConfig,
    run_position_modeling_workflow,
)


class PositionModelingWorkflowTests(unittest.TestCase):
    def _synthetic_data(self, n=120):
        rng = np.random.default_rng(7)
        pos = np.where(np.arange(n) % 2 == 0, "WR", "CB")
        forty = rng.normal(4.55, 0.1, size=n)
        vertical = rng.normal(35, 3, size=n)
        bench = rng.normal(18, 4, size=n)
        broad = rng.normal(120, 8, size=n)
        threecone = rng.normal(7.1, 0.2, size=n)
        shuttle = rng.normal(4.3, 0.15, size=n)
        wt = rng.normal(205, 15, size=n)
        career_year = rng.integers(1, 8, size=n)

        starts = np.maximum(
            0,
            25 + (vertical - 35) * 1.2 - (forty - 4.5) * 35 + rng.normal(0, 5, size=n),
        )
        av = np.maximum(0, 12 + starts * 0.35 + rng.normal(0, 4, size=n))
        snap = np.clip(0.2 + starts / 120 + rng.normal(0, 0.05, size=n), 0, 1)
        seasons = np.clip(career_year + rng.normal(0, 0.7, size=n), 0, 20)

        df = pd.DataFrame(
            {
                "Player": [f"P{i}" for i in range(n)],
                "NFL_id": np.arange(n),
                "Pos": pos,
                "combine_year": 2010 + (np.arange(n) % 10),
                "Ht": ["6-0"] * n,
                "Wt": wt,
                "40yd": forty,
                "Vertical": vertical,
                "Bench": bench,
                "Broad Jump": broad,
                "3Cone": threecone,
                "Shuttle": shuttle,
                "career_year": career_year,
                "starts": starts,
                "approximate_value": av,
                "snap_share": snap,
                "seasons_active": seasons,
            }
        )

        df.loc[df.index[::11], "Bench"] = np.nan
        df.loc[df.index[::13], "3Cone"] = np.nan
        return df

    def test_workflow_outputs_and_versions(self):
        df = self._synthetic_data()
        with tempfile.TemporaryDirectory() as td:
            outputs = run_position_modeling_workflow(
                df,
                output_dir=td,
                config=PositionModelingConfig(
                    bootstrap_iterations=30, min_group_size=20
                ),
            )

            for key in ["predictions", "feature_effects", "diagnostics", "calibration"]:
                self.assertIn(key, outputs)
                self.assertGreater(len(outputs[key]), 0)
                self.assertIn("heuristic_version", outputs[key].columns)
                self.assertIn("model_version", outputs[key].columns)

            self.assertTrue((Path(td) / "metadata.json").exists())
            self.assertTrue((Path(td) / "predictions.csv").exists())

            preds = outputs["predictions"]
            self.assertIn("pred_interval_lower", preds.columns)
            self.assertIn("pred_interval_upper", preds.columns)
            self.assertTrue(
                (preds["pred_interval_upper"] >= preds["pred_interval_lower"]).all()
            )


if __name__ == "__main__":
    unittest.main()
