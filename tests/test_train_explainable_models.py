import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.modeling.train_explainable_models import (
    _assert_no_forbidden_features,
    train_models,
)


class TrainExplainableModelsIdentityTests(unittest.TestCase):
    def test_assert_no_forbidden_features_raises_on_leakage(self):
        with self.assertRaisesRegex(ValueError, "Forbidden identity features"):
            _assert_no_forbidden_features(
                transformed_feature_names=["cat__Player_DB One", "num__forty_yard_sec_z"],
                forbidden_columns=["Player", "NFL_id"],
            )

    def test_training_excludes_identity_columns_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "input.csv"
            output_dir = tmp / "outputs"

            df = pd.DataFrame(
                {
                    "Player": [f"P{i}" for i in range(15)],
                    "NFL_id": list(range(100, 115)),
                    "record_id": list(range(200, 215)),
                    "dataset_split": ["train"] * 9 + ["val"] * 3 + ["test"] * 3,
                    "NFL_production_value": [float(i) for i in range(15)],
                    "forty_yard_sec_z": [0.1 * i for i in range(15)],
                    "bench_reps_z": [1.0 + 0.2 * i for i in range(15)],
                    "forty_yard_sec_missing": [0, 1, 0] * 5,
                }
            )
            df.to_csv(input_csv, index=False)

            train_models(input_csv=input_csv, output_dir=output_dir)

            manifest = json.loads((output_dir / "training_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["excluded_identity_columns"], ["NFL_id", "Player", "record_id"])
            self.assertNotIn("Player", manifest["model_feature_columns"])
            self.assertNotIn("NFL_id", manifest["model_feature_columns"])
            self.assertNotIn("record_id", manifest["model_feature_columns"])

            for model_name in ["ridge", "elastic_net", "decision_tree"]:
                explanation = pd.read_csv(output_dir / f"{model_name}_global_explanations.csv")
                features_lower = explanation["feature"].astype(str).str.lower()
                self.assertFalse(features_lower.str.contains("player").any())
                self.assertFalse(features_lower.str.contains("nfl_id").any())
                self.assertFalse(features_lower.str.contains("record_id").any())


if __name__ == "__main__":
    unittest.main()
