import tensorflow as tf

# Load test data
test_data = tf.data.experimental.load('../data/test_data')
X_test, y_test = test_data

# Load the trained model
model = tf.keras.models.load_model('../models/sentiment_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')
