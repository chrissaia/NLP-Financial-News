import tensorflow as tf
from src.model import create_model

# Load the preprocessed data
train_data = tf.data.experimental.load('../data/train_data')
X_train, y_train = train_data

# Get vocabulary size
vocab_size = 10000  # Assume the tokenizer has 10k words in vocab

# Create the model
model = create_model(vocab_size)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save the model
model.save('../models/sentiment_model.h5')
