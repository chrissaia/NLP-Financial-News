import unittest
import pandas as pd
from src.preprocess import preprocess_data, split_data

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        # Create a mock dataset
        self.data = pd.DataFrame({
            'headline': ['Stocks rise on tech rally', 'Oil prices fall after OPEC meeting', 'Tesla beats Q3 earnings estimates'],
            'sentiment': ['Positive', 'Negative', 'Positive']
        })
    
    def test_preprocess_data(self):
        # Test that preprocessing works correctly
        X, y = preprocess_data(self.data)
        
        # Check that the sequences are padded to the same length
        self.assertEqual(X.shape[1], 10, "Sequences should be padded to length 10")
        
        # Check that the labels are correctly encoded
        self.assertEqual(list(y), [1, 0, 1], "Sentiment labels should be correctly encoded as 1 (positive) and 0 (negative)")
    
    def test_split_data(self):
        # Test that splitting the data works correctly
        X, y = preprocess_data(self.data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Check that the split is correct
        self.assertEqual(len(X_train), 2, "Training set should contain 2 examples")
        self.assertEqual(len(X_test), 1, "Test set should contain 1 example")

if __name__ == '__main__':
    unittest.main()
