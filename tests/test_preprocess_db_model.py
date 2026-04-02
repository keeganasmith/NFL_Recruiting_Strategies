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


if __name__ == "__main__":
    unittest.main()
