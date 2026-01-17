import tensorflow as tf
import numpy as np

texts = [
    "Win money now",
    "Free gift card",
    "Call me tomorrow",
    "Let's have lunch",
    "Congratulations you won",
    "Meeting at 10 am",
    "Claim your free prize",
    "Are you coming today"
]

labels = np.array([1, 1, 0, 0, 1, 0, 1, 0])

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

model.fit(padded_sequences, labels, epochs=50)

test_text = ["You have won a free prize"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = tf.keras.preprocessing.sequence.pad_sequences(
    test_seq,
    maxlen=padded_sequences.shape[1],
    padding='post'
)

prediction = model.predict(test_pad)

if prediction >= 0.5:
    print("Spam Email")
else:
    print("Not Spam Email")
