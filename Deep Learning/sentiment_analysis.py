import tensorflow as tf
import numpy as np

texts = [
    "I love this movie",
    "This film was fantastic",
    "Amazing experience",
    "I hated this movie",
    "Worst movie ever",
    "Very boring and bad",
    "I enjoyed the film",
    "Terrible acting"
]

labels = np.array([1, 1, 1, 0, 0, 0, 1, 0])

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    padding='post'
)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(padded_sequences, labels, epochs=60)

test_reviews = [
    "I really loved this film",
    "This was the worst movie"
]

test_seq = tokenizer.texts_to_sequences(test_reviews)
test_pad = tf.keras.preprocessing.sequence.pad_sequences(
    test_seq,
    maxlen=padded_sequences.shape[1],
    padding='post'
)

predictions = model.predict(test_pad)

for review, pred in zip(test_reviews, predictions):
    sentiment = "Positive" if pred >= 0.5 else "Negative"
    print(review, "→", sentiment)
