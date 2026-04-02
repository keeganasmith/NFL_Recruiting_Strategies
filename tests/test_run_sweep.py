import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.run_sweep import _write_best_scored_players


class RunSweepTests(unittest.TestCase):
    def test_write_best_scored_players_creates_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "sweep_outputs"
            run_dir = output_root / "variant_001"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"Player": "A", "NFL_production_value": 12.0},
                    {"Player": "B", "NFL_production_value": 7.5},
                ]
            ).to_csv(run_dir / "scored_players.csv", index=False)

            comparison_df = pd.DataFrame(
                [
                    {"variant_id": "variant_001", "run_directory": str(run_dir), "objective_score": 0.9},
                    {"variant_id": "variant_002", "run_directory": str(output_root / "variant_002"), "objective_score": 0.7},
                ]
            )

            artifact_name, best_variant_id = _write_best_scored_players(output_root, comparison_df)
            artifact_path = output_root / artifact_name

            self.assertEqual(best_variant_id, "variant_001")
            self.assertTrue(artifact_path.exists())
            self.assertIn("NFL_production_value", pd.read_csv(artifact_path).columns)

    def test_write_best_scored_players_requires_nfl_production_value_column(self):
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "sweep_outputs"
            run_dir = output_root / "variant_001"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"Player": "A", "not_score": 1.0}]).to_csv(run_dir / "scored_players.csv", index=False)

            comparison_df = pd.DataFrame(
                [{"variant_id": "variant_001", "run_directory": str(run_dir), "objective_score": 0.9}]
            )

            with self.assertRaises(ValueError):
                _write_best_scored_players(output_root, comparison_df)


if __name__ == "__main__":
    unittest.main()
