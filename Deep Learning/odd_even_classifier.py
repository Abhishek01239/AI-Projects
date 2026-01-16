import tensorflow as tf
import numpy as np

x = np.array([0,1,2,3,4,5,6,7,8,9], dtype = float)
y = np.array([0,1,0,1,0,1,0,1,0,1], dtype = float)

x = x/10.0

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation = 'relu', input_shape= [1]),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(1,activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

model.fit(x,y, epochs = 500)

num = 7
scaled_num = num/10.0

prediction = model.predict(np.array([[scaled_num]]))

if prediction >=0.5:
    print(num, "is odd")
else:
    print(num, "is even")