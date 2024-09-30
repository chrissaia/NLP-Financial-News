import unittest
import tensorflow as tf
from src.model import create_model
from src.preprocess import preprocess_data, split_data
import pandas as pd

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset
        self.data = pd.DataFrame({
            'headline': ['Stocks rise on tech rally', 'Oil prices fall after OPEC meeting', 'Tesla beats Q3 earnings estimates'],
            'sentiment': ['Positive', 'Negative', 'Positive']
        })
        X, y = preprocess_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(X, y)
        self.vocab_size = 1000
        
        # Train a model to evaluate
        self.model = create_model(self.vocab_size)
        self.model.fit(self.X_train, self.y_train, epochs=1)
    
    def test_model_evaluation(self):
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        
        # Ensure accuracy is returned and within the expected range
        self.assertGreaterEqual(accuracy, 0, "Accuracy should be non-negative")
        self.assertLessEqual(accuracy, 1, "Accuracy should not be greater than 1")

if __name__ == '__main__':
    unittest.main()
