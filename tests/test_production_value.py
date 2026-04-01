import unittest
from pathlib import Path

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.production_value import (
    compute_production_value,
    compute_production_value_batch,
    load_production_value_config,
)


class ProductionValueTests(unittest.TestCase):
    def setUp(self):
        self.config = load_production_value_config()
        self.base_row = {
            "Player": "Base Player",
            "Pos": "WR",
            "career_year": 4,
            "starts": 20,
            "approximate_value": 18,
            "snap_share": 0.55,
            "seasons_active": 4,
        }

    def test_single_row_and_batch_match(self):
        single = compute_production_value(self.base_row, self.config)
        batch = (
            compute_production_value_batch(pd.DataFrame([self.base_row]), self.config)
            .iloc[0]
            .to_dict()
        )
        self.assertAlmostEqual(
            single["production_value"], batch["production_value"], places=12
        )
        self.assertEqual(single["heuristic_version"], self.config["version"])

    def test_monotonicity_starts_increase(self):
        low = dict(self.base_row)
        high = dict(self.base_row)
        high["starts"] = low["starts"] + 25
        low_score = compute_production_value(low, self.config)["production_value"]
        high_score = compute_production_value(high, self.config)["production_value"]
        self.assertGreaterEqual(high_score, low_score)

    def test_sensitivity_approximate_value_increase(self):
        rows = [dict(self.base_row), dict(self.base_row)]
        rows[1]["approximate_value"] = rows[0]["approximate_value"] + 30
        out = compute_production_value_batch(pd.DataFrame(rows), self.config)
        scores = out["production_value"].tolist()
        self.assertGreaterEqual(scores[1], scores[0])

    def test_position_override_applies(self):
        wr_row = dict(self.base_row)
        qb_row = dict(self.base_row)
        qb_row["Pos"] = "QB"
        qb_row["Player"] = "QB Player"
        out = compute_production_value_batch(
            pd.DataFrame([wr_row, qb_row]), self.config
        )
        wr_score = out.loc[out["Pos"] == "WR", "production_value"].iloc[0]
        qb_score = out.loc[out["Pos"] == "QB", "production_value"].iloc[0]
        self.assertNotEqual(qb_score, wr_score)

    def test_input_validation_missing_columns(self):
        bad_df = pd.DataFrame([{"Pos": "WR", "starts": 2}])
        with self.assertRaises(ValueError):
            compute_production_value_batch(bad_df, self.config)

    def test_input_validation_out_of_range(self):
        bad_row = dict(self.base_row)
        bad_row["snap_share"] = 1.8
        with self.assertRaises(ValueError):
            compute_production_value(bad_row, self.config)


if __name__ == "__main__":
    unittest.main()
