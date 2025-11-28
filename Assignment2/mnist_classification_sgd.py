import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib


# DATA
print("[INFO] Loading MNIST training data...")
df = pd.read_csv("Q2\\mnist_train.csv")

X = df.iloc[:, 1:]  
y = df.iloc[:, 0]  

print(f"[INFO] Total training samples: {len(X)}")
print(f"[INFO] Features per image: {X.shape[1]}")
print(f"[INFO] Classes: {sorted(y.unique())}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(f"[INFO] Training samples: {len(X_train)}")
print(f"[INFO] Testing samples: {len(X_test)}")


# MODEL
print("[INFO] Training SGD Classifier model...")
model = SGDClassifier(max_iter=100, random_state=42)
model.fit(X_train, y_train)


# EVALUATE
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'\n[INFO] Accuracy = {accuracy * 100:.2f}%')

joblib.dump(model, 'mnist_sgd_model.z')
joblib.dump(sc, 'mnist_scaler_sgd.z')
print("[INFO] Model saved as 'mnist_sgd_model.z'")
