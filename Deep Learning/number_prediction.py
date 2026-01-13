import tensorflow as tf
import numpy as np

# Data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train
model.fit(x, y, epochs=500)

# Predict
print("Prediction for 7:")
print(model.predict(np.array([[7]], dtype=float)))
