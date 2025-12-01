### 
## cluster_maker - test_preprocessing.py
## Athul
## December 2025
###

import unittest
import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):

    # Test that select_features raises KeyError when requested columns are missing.
    # This detects silent failures where clustering runs on unintended columns.
    def test_select_features_missing_column(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        })
        with self.assertRaises(KeyError):
            select_features(df, ["x", "z"])  # 'z' is missing

    # Test that standardise_features correctly transforms data to zero mean and unit variance.
    # This ensures that scaling is applied properly before clustering.
    def test_standardise_features_scaling(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_scaled = standardise_features(X)
        self.assertTrue(np.allclose(X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-7))
        self.assertTrue(np.allclose(X_scaled.std(axis=0), [1.0, 1.0], atol=1e-7))

    # Test that select_features raises TypeError when selected columns are non-numeric.
    # This prevents downstream crashes in clustering due to invalid feature types.
    def test_select_features_non_numeric_column(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": ["a", "b", "c"],  # non-numeric
        })
        with self.assertRaises(TypeError):
            select_features(df, ["x", "y"])


if __name__ == "__main__":
    unittest.main()