import numpy as np
import pandas as pd
import re
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from sklearn.model_selection import train_test_split

# ---------- 1. LOAD DATA ----------
print("Loading data...")
data = pd.read_csv("Reviews.csv")  # Make sure Reviews.csv is in same folder

# Keep only useful columns
data = data[['Text', 'Score']]

# Remove missing values
data.dropna(inplace=True)

# Convert scores into binary sentiment: 1 (Positive), 0 (Negative)
data['Sentiment'] = data['Score'].apply(lambda x: 1 if x > 3 else 0)

print("Class distribution:")
print(data['Sentiment'].value_counts())

# ---------- 2. CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

print("Cleaning text...")
data['Cleaned_Text'] = data['Text'].apply(clean_text)

# ---------- 3. TOKENIZE & PAD ----------
max_words = 10000
max_len = 200

print("Tokenizing text...")
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Cleaned_Text'])

sequences = tokenizer.texts_to_sequences(data['Cleaned_Text'])
X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

y = data['Sentiment'].values

# ---------- 4. TRAIN / TEST SPLIT ----------
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 5. BUILD GRU MODEL ----------
print("Building model...")
model_gru = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    GRU(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model_gru.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ---------- 6. TRAIN MODEL ----------
print("Training model...")
history_gru = model_gru.fit(
    X_train, y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.2
)

# ---------- 7. EVALUATE MODEL ----------
print("Evaluating model...")
loss, acc = model_gru.evaluate(X_test, y_test, verbose=0)
print(f"GRU Test Accuracy: {acc:.4f}")

# ---------- 8. SAVE TOKENIZER & MODEL ----------
print("Saving tokenizer and model...")

# Save tokenizer
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save model
model_gru.save("sentiment_gru.h5")

print("Done! Files saved: tokenizer.pickle and sentiment_gru.h5")
