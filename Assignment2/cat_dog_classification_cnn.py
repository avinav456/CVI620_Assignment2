import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import layers, Sequential
import matplotlib.pyplot as plt
import os
import joblib


# DATA
data_list = []
label_list = []

for i, address in enumerate(glob.glob("Q1\\train\\*\\*")):
    image = cv2.imread(address)
    image = cv2.resize(image, (32,32))
    image = image/255

    data_list.append(image)

    label_list.append(os.path.basename(os.path.dirname(address)))

    if i%200 == 0:
        print(f'[INFO] {i} images processed!')

X = np.array(data_list)
y = np.array(label_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_test)

# MODEL
nn = Sequential([
        layers.Conv2D(8, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(2, activation='softmax')
])


nn.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

H = nn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)


# EVALUATE
loss, accuracy = nn.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {accuracy * 100:.2f}%')
print(f'Test Loss: {loss:.4f}')

nn.save('cat_dog_cnn_model.h5')
joblib.dump(le, 'label_encoder.z')
print('\n[INFO] Model saved as cat_dog_cnn_model.h5')
print('[INFO] Label encoder saved as label_encoder.z')

plt.plot(H.history['accuracy'], label='train accuracy')
plt.plot(H.history['val_accuracy'], label='test accuracy')
plt.plot(H.history['loss'], label='loss')
plt.plot(H.history['val_loss'], label='test loss')
plt.legend()

plt.show()