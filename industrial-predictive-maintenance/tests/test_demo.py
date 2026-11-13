import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import FEATURES, add_past_only_features, make_synthetic_data
import pandas as pd


class DemoTests(unittest.TestCase):
    def test_reproducible(self):
        a, b = make_synthetic_data(), make_synthetic_data()
        pd.testing.assert_frame_equal(a, b)

    def test_past_only(self):
        df = make_synthetic_data(n_machines=2, n_hours=60)
        earlier = df[df.hour <= 29].copy()
        changed = df.copy()
        changed.loc[changed.hour > 29, "temperature"] = 9999
        again = add_past_only_features(changed)[lambda x: x.hour <= 29]
        pd.testing.assert_series_equal(
            earlier["temp_mean_6"].reset_index(drop=True),
            again["temp_mean_6"].reset_index(drop=True),
            check_names=False
        )

    def test_expected_schema(self):
        data = make_synthetic_data()
        self.assertTrue(all(c in data.columns for c in FEATURES))
        self.assertFalse(data[FEATURES].isna().any().any())
        self.assertTrue(set(data.failure_next_6h.unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
