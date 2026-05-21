import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

X = np.load('X.npy')
y = np.load('y.npy')
X = np.expand_dims(X, axis=-1)

model = models.Sequential([
    layers.Input(shape=(128, 87, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=15)
model.save('glass_shield.h5')
print("Model saved as glass_shield.h5")