import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import layers, models

# Step 1: Load Dataset
max_features = 10000  # Use top 10k words
maxlen = 200          # Limit review length to 200 words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print("Training samples:", len(x_train))
print("Test samples:", len(x_test))

# Step 2: Pad sequences (make all reviews same length)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Step 3: Build the LSTM Model
model = models.Sequential([
    layers.Embedding(max_features, 128, input_length=maxlen),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

# Step 4: Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Train the Model
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=3,
                    validation_split=0.2)

# Step 6: Evaluate on Test Set
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# Step 7: Predict Sentiment on a Custom Review
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Take one test review and decode it
sample_review = decode_review(x_test[0])
print("\nSample Review:")
print(sample_review)

prediction = model.predict(x_test[:1])[0][0]
sentiment = "Positive 😄" if prediction > 0.5 else "Negative 😞"
print(f"\nPredicted Sentiment: {sentiment}")
