import unittest
import pandas as pd
from src.preprocessing import preprocess_data

class TestPreprocessing(unittest.TestCase):

    def test_preprocess_data(self):
        # Sample data for testing
        sample_data = {
            'feature1': [1, 2, None],
            'feature2': ['A', 'B', 'C'],
            'target': [0, 1, 1]
        }
        df = pd.DataFrame(sample_data)
        
        processed_data = preprocess_data(df)

        # Check if missing values are handled
        self.assertFalse(processed_data.isnull().values.any(), "There are still missing values")

        # Check if the shape is correct after preprocessing
        self.assertEqual(processed_data.shape[0], df.shape[0], "Number of rows should remain the same after preprocessing")

if __name__ == '__main__':
    unittest.main()
