import unittest
import tensorflow as tf
from src.model import create_model
from src.preprocess import preprocess_data, split_data
import pandas as pd

class TestTraining(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset
        self.data = pd.DataFrame({
            'headline': ['Stocks rise on tech rally', 'Oil prices fall after OPEC meeting', 'Tesla beats Q3 earnings estimates'],
            'sentiment': ['Positive', 'Negative', 'Positive']
        })
        X, y = preprocess_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(X, y)
        self.vocab_size = 1000
    
    def test_model_training(self):
        model = create_model(self.vocab_size)
        
        # Test that the model can be trained without errors
        history = model.fit(self.X_train, self.y_train, epochs=1, validation_split=0.2)
        
        # Check if the model improved after training (accuracy > 0)
        final_accuracy = history.history['accuracy'][-1]
        self.assertGreater(final_accuracy, 0, "Model accuracy should be greater than 0 after training")

if __name__ == '__main__':
    unittest.main()
