"""Offline tests using invented fixture data, never UCI benchmark results."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai4i_baseline import FEATURES, EXCLUDED, make_splits, prepare_features, run


def fixture(n=200):
    rng = np.random.default_rng(73)
    y = np.tile([0, 0, 0, 0, 1], n // 5)
    return pd.DataFrame({
        "UDI": np.arange(n),
        "Product ID": [f"L{i}" for i in range(n)],
        "Type": np.tile(["L", "M", "H", "L", "M"], n // 5),
        "Air temperature [K]": 298 + rng.normal(0, 1, n) + 2 * y,
        "Process temperature [K]": 308 + rng.normal(0, 1, n) + 2 * y,
        "Rotational speed [rpm]": 1500 + rng.normal(0, 30, n),
        "Torque [Nm]": 40 + rng.normal(0, 4, n) + 9 * y,
        "Tool wear [min]": rng.uniform(0, 200, n),
        "Machine failure": y,
        "TWF": y, "HDF": y, "PWF": y, "OSF": y, "RNF": y,
    })


class Ai4iTests(unittest.TestCase):
    def test_leakage_columns_not_used(self):
        df = fixture()
        x, y = prepare_features(df)
        self.assertEqual(list(x.columns), FEATURES)
        self.assertTrue(set(EXCLUDED).isdisjoint(x.columns))
        altered = df.copy()
        for field in EXCLUDED:
            altered[field] = "REDACTED"
        new_x, new_y = prepare_features(altered)
        pd.testing.assert_frame_equal(x, new_x)
        pd.testing.assert_series_equal(y, new_y)

    def test_split_is_disjoint_and_stratified(self):
        _, y = prepare_features(fixture())
        train, val, test = make_splits(y)
        self.assertEqual((len(train), len(val), len(test)), (120, 40, 40))
        self.assertEqual(len(set(train) | set(val) | set(test)), 200)
        self.assertEqual(
            (int(y.iloc[train].sum()), int(y.iloc[val].sum()), int(y.iloc[test].sum())),
            (24, 8, 8),
        )

    def test_invalid_rows_are_rejected(self):
        df = fixture()
        df.loc[0, "Type"] = "unknown"
        with self.assertRaises(ValueError):
            prepare_features(df)
        df = fixture()
        df.loc[0, "Torque [Nm]"] = np.nan
        with self.assertRaises(ValueError):
            prepare_features(df)

    def test_fixture_pipeline_exports(self):
        with tempfile.TemporaryDirectory() as path:
            csv = Path(path) / "invented_fixture.csv"
            target = Path(path) / "output"
            fixture().to_csv(csv, index=False)
            result = run(csv, output_dir=target, n_estimators=20)
            self.assertEqual(result["counts"]["test"], 40)
            for name in ("model.joblib", "metrics.json", "test_predictions.csv"):
                self.assertTrue((target / name).is_file())
            self.assertGreaterEqual(result["test_validation_selected"]["recall"], 0)


if __name__ == "__main__":
    unittest.main()
