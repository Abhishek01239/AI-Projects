import tensorflow as tf
import numpy as np

x = np.array([1,2,3,4,5,6,7,8], dtype = float)
y = np.array([20,30,40, 55, 65,75,85,95], dtype = float)

x = x/10.0
y = y/100.0

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation = 'relu', input_shape = [1]),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer = 'adam',
    loss = 'mean_squared_error'
)

history = model.fit(x, y, epochs = 800, validation_split = 0.2)

prediction = model.predict(np.array([[0.6]]))
print("Predicted marks", prediction *100)

