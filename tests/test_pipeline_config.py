import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.config import load_experiment_config, load_heuristic_config, validate_heuristic_inputs


class PipelineConfigTests(unittest.TestCase):
    def test_load_heuristic_yaml(self):
        cfg = load_heuristic_config("configs/heuristics/nfl_production_value_baseline.yaml")
        self.assertEqual(cfg.heuristic_id, "weighted_nfl_production_value")
        self.assertIn("defensive_interceptions", cfg.feature_weights)

    def test_invalid_weight_range_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.yaml"
            p.write_text(
                """
heuristic_id: weighted_nfl_production_value
feature_weights:
  defensive_totalTackles: 9999
  defensive_sacks: 2
  defensive_interceptions: 3
  defensive_passesDefended: 1.5
  defensive_gamesPlayed: 0.5
"""
            )
            with self.assertRaises(ValueError):
                load_heuristic_config(p)

    def test_validate_heuristic_inputs_missing_columns(self):
        cfg = load_heuristic_config("configs/heuristics/nfl_production_value_baseline.yaml")
        with self.assertRaises(ValueError):
            validate_heuristic_inputs(pd.DataFrame([{"Pos": "CB"}]), cfg)

    def test_load_experiment_yaml(self):
        exp = load_experiment_config("configs/experiments/default_experiment.yaml")
        self.assertEqual(exp.experiment_id, "default_heuristic_sweep")
        self.assertGreaterEqual(len(exp.variants), 1)


if __name__ == "__main__":
    unittest.main()
