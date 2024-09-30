import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Load the dataset
data = pd.read_csv('../data/data.csv')

# Preprocess the data
def preprocess_data(data):
    data['label'] = LabelEncoder().fit_transform(data['sentiment'])
    
    # Tokenize the text data
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(data['headline'])
    sequences = tokenizer.texts_to_sequences(data['headline'])
    
    # Padding sequences to the same length
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    
    return padded_sequences, data['label']

# Split the data
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(X, y)

# Save the preprocessed data for training
tf.data.experimental.save((X_train, y_train), '../data/train_data')
tf.data.experimental.save((X_test, y_test), '../data/test_data')
