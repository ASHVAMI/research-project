import unittest
from src.model import build_model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class TestModel(unittest.TestCase):

    def test_build_model(self):
        # Sample data for testing
        sample_data = {
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'target': [0, 1, 0, 1]
        }
        df = pd.DataFrame(sample_data)

        model, X_test, y_test = build_model(df)

        # Check if the model is an instance of RandomForestClassifier
        self.assertIsInstance(model, RandomForestClassifier, "Model should be a RandomForestClassifier")

        # Check if X_test and y_test are non-empty
        self.assertGreater(len(X_test), 0, "X_test should not be empty")
        self.assertGreater(len(y_test), 0, "y_test should not be empty")

if __name__ == '__main__':
    unittest.main()
