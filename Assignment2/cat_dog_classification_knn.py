import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import glob
import joblib
import os


# DATA
data_list = []
label_list = []

print("[INFO] Loading training data...")
for i, address in enumerate(glob.glob("Q1\\train\\*\\*")):
    image = cv2.imread(address)
    if image is None:
        continue
    
    image = cv2.resize(image, (32, 32))
    image = image.flatten()
    image = image / 255.0
    
    data_list.append(image)
    
    label = os.path.basename(os.path.dirname(address))
    label_list.append(label)
    
    if i % 200 == 0:
        print(f'[INFO] {i} images processed!')

print(f"[INFO] Total images loaded: {len(data_list)}")



X = np.array(data_list)
y = np.array(label_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

joblib.dump(sc, 'scaler_knn.z')

print(f"[INFO] Training samples: {len(X_train)}")
print(f"[INFO] Testing samples: {len(X_test)}")


# MODEL
print("[INFO] Training KNN model...")
best_k = 5
best_accuracy = 0

for k in [3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"K={k}, Accuracy={accuracy * 100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"\n[INFO] Best K value: {best_k}")

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)


# EVALUATE
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'\n[INFO] Final Accuracy = {accuracy * 100:.2f}%')

joblib.dump(model, 'cat_dog_knn_model.z')
print("[INFO] Model saved as 'cat_dog_knn_model.z'")