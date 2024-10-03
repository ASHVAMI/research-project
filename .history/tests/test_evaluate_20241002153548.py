import unittest
from src.evaluate import evaluate_model
from src.model import build_model
import pandas as pd

class TestEvaluate(unittest.TestCase):

    def test_evaluate_model(self):
        # Sample data for testing
        sample_data = {
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'target': [0, 1, 0, 1]
        }
        df = pd.DataFrame(sample_data)

        # Build the model
        model, X_test, y_test = build_model(df)

        # Evaluate the model
        accuracy = evaluate_model(model, X_test, y_test)

        # Check if the accuracy is a float and between 0 and 1
        self.assertIsInstance(accuracy, float, "Accuracy should be a float")
        self.assertGreaterEqual(accuracy, 0, "Accuracy should be greater than or equal to 0")
        self.assertLessEqual(accuracy, 1, "Accuracy should be less than or equal to 1")

if __name__ == '__main__':
    unittest.main()
