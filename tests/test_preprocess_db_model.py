import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.preprocess_db_model import SplitConfig, preprocess_db_model_dataset


class PreprocessDbModelTests(unittest.TestCase):
    def test_preprocess_writes_manifest_and_splits(self):
        with tempfile.TemporaryDirectory() as td:
            out_csv = Path(td) / "processed.csv"
            manifest = Path(td) / "feature_manifest.json"

            df, feature_manifest = preprocess_db_model_dataset(
                output_csv=out_csv,
                manifest_path=manifest,
                split_config=SplitConfig(seed=42, mode="random", val_size=0.2, test_size=0.2),
            )

            self.assertTrue(out_csv.exists())
            self.assertTrue(manifest.exists())
            self.assertGreater(len(df), 0)
            self.assertIn("dataset_split", df.columns)
            self.assertTrue(set(df["dataset_split"].unique()).issubset({"train", "val", "test"}))
            self.assertGreater(len(feature_manifest["model_input_features"]), 0)
            self.assertIn("NFL_production_value", df.columns)

    def test_draft_year_split(self):
        with tempfile.TemporaryDirectory() as td:
            out_csv = Path(td) / "processed.csv"
            manifest = Path(td) / "feature_manifest.json"
            df, _ = preprocess_db_model_dataset(
                output_csv=out_csv,
                manifest_path=manifest,
                split_config=SplitConfig(mode="draft_year", seed=42),
            )
            self.assertIn("split_val_start_year", df.columns)
            self.assertIn("split_test_start_year", df.columns)

    def test_preprocess_collapses_duplicate_player_rows(self):
        with tempfile.TemporaryDirectory() as td:
            input_csv = Path(td) / "input.csv"
            out_csv = Path(td) / "processed.csv"
            manifest = Path(td) / "feature_manifest.json"

            pd.DataFrame(
                [
                    {
                        "Player": "DB One",
                        "NFL_id": 1,
                        "Pos": "CB",
                        "combine_year": 2020,
                        "season_year": 2020,
                        "Ht": "6-0",
                        "Wt": 190,
                        "NFL_production_value": 10.0,
                    },
                    {
                        "Player": "DB One",
                        "NFL_id": 1,
                        "Pos": "CB",
                        "combine_year": 2020,
                        "season_year": 2021,
                        "Ht": "6-0",
                        "Wt": 190,
                        "NFL_production_value": 5.0,
                    },
                    {
                        "Player": "DB Two",
                        "NFL_id": 2,
                        "Pos": "S",
                        "combine_year": 2020,
                        "season_year": 2020,
                        "Ht": "5-11",
                        "Wt": 200,
                        "NFL_production_value": 7.0,
                    },
                ]
            ).to_csv(input_csv, index=False)

            df, _ = preprocess_db_model_dataset(
                input_csv=input_csv,
                output_csv=out_csv,
                manifest_path=manifest,
                split_config=SplitConfig(seed=42, mode="random", val_size=0.2, test_size=0.2),
            )
            self.assertEqual(len(df), 2)

            collapsed_score = float(df.loc[df["NFL_id"] == 1, "NFL_production_value"].iloc[0])
            self.assertEqual(collapsed_score, 15.0)


if __name__ == "__main__":
    unittest.main()
