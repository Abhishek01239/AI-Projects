import tensorflow as tf
import numpy as np

# Dataset
x = np.array([500, 800, 1000, 1200, 1500, 1800, 2000], dtype=float)
y = np.array([25, 40, 50, 60, 75, 90, 100], dtype=float)

# Feature scaling
x = x / 1000.0
y = y / 100.0

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(x, y, epochs=1000)

# Predict
prediction = model.predict(np.array([[1.4]]))
print("Predicted house price (in lakhs):", prediction * 100)
