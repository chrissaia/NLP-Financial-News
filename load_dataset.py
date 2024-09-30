import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load dataset
data = pd.read_csv('news_data.csv')

# Example structure of the dataset
# | text                           | label  |
# |---------------------------------|--------|
# | "The market is doing great!"    | positive|
# | "The company faces challenges." | negative|

# Preprocessing
nlp = spacy.load('en_core_web_md')
tokenizer = Tokenizer()

# Encode labels (positive = 0, negative = 1, neutral = 2)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Tokenization and padding
X = data['text'].values
y = to_categorical(data['label'], num_classes=3)

# Tokenizing the sentences
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Padding sequences to ensure all inputs are of the same length
max_len = 100  # Max number of words per sentence
X_padded = pad_sequences(X_seq, maxlen=max_len)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
