import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.evaluation.heuristic_objective import evaluate_heuristic
from pipeline.run_sweep import _build_auto_grid_variants


class HeuristicObjectiveTests(unittest.TestCase):
    def test_evaluate_heuristic_returns_finite_objective(self):
        scored = pd.DataFrame(
            {
                "NFL_production_value": [1.0, 2.0, 3.0, 4.0, 5.0],
                "defensive_gamesPlayed": [10, 20, 30, 40, 50],
                "defensive_totalTackles": [5, 10, 20, 35, 45],
                "defensive_sacks": [0, 1, 1, 2, 3],
                "defensive_interceptions": [0, 0, 1, 2, 3],
                "defensive_passesDefended": [1, 2, 4, 5, 7],
            }
        )
        ranking = pd.DataFrame([{"spearman_rank_corr": 0.3, "top_n_overlap_rate": 0.4}])

        out = evaluate_heuristic(scored_df=scored, ranking_df=ranking)

        self.assertGreater(out.proxy_spearman, 0.0)
        self.assertGreater(out.objective_score, 0.0)

    def test_auto_grid_writes_variant_configs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp_path = root / "exp.yaml"
            base_cfg = root / "base.yaml"
            base_cfg.write_text(
                """
heuristic_id: weighted_nfl_production_value
feature_weights:
  defensive_totalTackles: 1.0
  defensive_sacks: 2.0
  defensive_interceptions: 3.0
  defensive_passesDefended: 1.5
  defensive_gamesPlayed: 0.5
thresholds: {}
role_overrides: {}
"""
            )
            exp_path.write_text("experiment_id: test\nheuristic_variants: [{id: a, heuristic_config: b}]\n")

            variants = _build_auto_grid_variants(
                experiment_config_path=exp_path,
                experiment_cfg={
                    "auto_grid": {
                        "base_heuristic_config": "base.yaml",
                        "output_config_dir": str(root / "configs" / "heuristics"),
                        "variant_prefix": "unit",
                        "feature_weights": {
                            "defensive_totalTackles": [0.9, 1.0],
                            "defensive_sacks": [1.8],
                        },
                    }
                },
            )

            self.assertEqual(len(variants), 2)
            self.assertTrue((root / "configs" / "heuristics" / "unit_001.yaml").exists())
            self.assertTrue((root / "configs" / "heuristics" / "unit_002.yaml").exists())


if __name__ == "__main__":
    unittest.main()
