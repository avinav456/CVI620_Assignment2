import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import layers, Sequential
import matplotlib.pyplot as plt
import joblib


# DATA
print("[INFO] Loading MNIST training data...")
df = pd.read_csv("Q2\\mnist_train.csv")

X = df.iloc[:, 1:].values  
y = df.iloc[:, 0].values   

print(f"[INFO] Total training samples: {len(X)}")
print(f"[INFO] Features per image: {X.shape[1]}")
print(f"[INFO] Classes: {sorted(np.unique(y))}")

X = X.reshape(-1, 28, 28, 1)
X = X / 255.0 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"[INFO] Training samples: {len(X_train)}")
print(f"[INFO] Testing samples: {len(X_test)}")


# MODEL
nn = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

nn.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

print(nn.summary())

print("[INFO] Training CNN model...")
H = nn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)


# EVALUATE
loss, accuracy = nn.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {accuracy * 100:.2f}%')
print(f'Test Loss: {loss:.4f}')

nn.save('mnist_cnn_model.h5')
print("[INFO] Model saved as 'mnist_cnn_model.h5'")

plt.plot(H.history['accuracy'], label='train accuracy')
plt.plot(H.history['val_accuracy'], label='test accuracy')
plt.plot(H.history['loss'], label='loss')
plt.plot(H.history['val_loss'], label='test loss')
plt.legend()

plt.show()
