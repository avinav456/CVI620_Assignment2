import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
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


print(f"[INFO] Training samples: {len(X_train)}")
print(f"[INFO] Testing samples: {len(X_test)}")


# MODEL
print("[INFO] Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


# EVALUATE
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'\n[INFO] Accuracy = {accuracy * 100:.2f}%')

joblib.dump(model, 'cat_dog_lr_model.z')
print("[INFO] Model saved as 'cat_dog_lr_model.z'")