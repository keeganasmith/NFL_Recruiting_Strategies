import unittest

import pandas as pd

from pipeline.features.preprocessing import prepare_features, restrict_to_early_career


class PipelinePreprocessingTests(unittest.TestCase):
    def test_restrict_to_first_four_career_years(self):
        df = pd.DataFrame(
            {
                "Player": ["A", "A", "A", "A", "A"],
                "combine_year": [2019, 2019, 2019, 2019, 2019],
                "season_year": [2019, 2020, 2021, 2022, 2023],
            }
        )
        out = restrict_to_early_career(df, max_career_year=4)
        self.assertEqual(len(out), 4)
        self.assertEqual(out["career_year"].max(), 4)

    def test_prepare_features_applies_early_career_filter(self):
        df = pd.DataFrame(
            {
                "Player": ["B", "B"],
                "combine_year": [2020, 2020],
                "season_year": [2020, 2025],
                "defensive_totalTackles": [10, 999],
            }
        )
        out = prepare_features(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out["defensive_totalTackles"].iloc[0]), 10.0)


if __name__ == "__main__":
    unittest.main()
